import io
from pathlib import Path

import fitz  # PyMuPDF
import numpy as np
import streamlit as st
import cv2
from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

st.set_page_config(page_title="MATCH | Ambalaj PDF Karşılaştırma", page_icon="🔎", layout="wide")

st.title("MATCH")
st.caption("Ambalaj tasarımı • PDF karşılaştırma prototipi")
st.info(
    "Bu MVP görsel farkları bulmaya yardımcı olur; baskıya uygunluk veya tüm hataların yakalandığı garantisi vermez. "
    "Son kontrol yetkili repro/prepress uzmanında kalmalıdır."
)

with st.sidebar:
    st.header("Karşılaştırma ayarları")
    dpi = st.select_slider("Görüntü çözünürlüğü (DPI)", options=[100, 150, 200, 250], value=150)
    threshold = st.slider("Fark hassasiyeti", min_value=5, max_value=80, value=25, help="Düşük değer daha fazla küçük fark gösterebilir.")
    min_area = st.slider("En küçük fark alanı (piksel)", min_value=10, max_value=1000, value=100)
    align_images = st.checkbox("Küçük kaymaları otomatik hizalamayı dene", value=False)
    st.caption("Otomatik hizalama gerçek yerleşim kaymalarını gizleyebilir. Kritik kontrollerde kapalı tutun.")

col_a, col_b = st.columns(2)
with col_a:
    original_file = st.file_uploader("1. Müşteri orijinal PDF", type=["pdf"], key="original")
with col_b:
    repro_file = st.file_uploader("2. Repro PDF", type=["pdf"], key="repro")

def pdf_pages(file_bytes):
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    return doc

def render_page(doc, index, dpi):
    page = doc.load_page(index)
    pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72), alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    return img, page.rect.width, page.rect.height

def fit_to_common_canvas(img_a, img_b):
    # Center each image on the same white canvas without stretching artwork.
    h = max(img_a.shape[0], img_b.shape[0])
    w = max(img_a.shape[1], img_b.shape[1])
    def place(img):
        canvas_img = np.full((h, w, 3), 255, dtype=np.uint8)
        y = (h - img.shape[0]) // 2
        x = (w - img.shape[1]) // 2
        canvas_img[y:y+img.shape[0], x:x+img.shape[1]] = img
        return canvas_img
    return place(img_a), place(img_b)

def make_diff(img_a, img_b, threshold, min_area, align_images):
    a, b = fit_to_common_canvas(img_a, img_b)
    alignment_note = "Hizalama kapalı"
    if align_images:
        try:
            gray_a = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
            gray_b = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY)
            warp = np.eye(2, 3, dtype=np.float32)
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 80, 1e-5)
            cv2.findTransformECC(gray_a, gray_b, warp, cv2.MOTION_TRANSLATION, criteria)
            b = cv2.warpAffine(b, warp, (b.shape[1], b.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255))
            alignment_note = "Otomatik öteleme hizalaması denendi"
        except Exception:
            alignment_note = "Otomatik hizalama yapılamadı; hizalama uygulanmadı"
    ga = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
    gb = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY)
    delta = cv2.absdiff(ga, gb)
    _, mask = cv2.threshold(delta, threshold, 255, cv2.THRESH_BINARY)
    # Reduce isolated rendering noise while preserving larger differences.
    kernel = np.ones((2, 2), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions = []
    overlay = b.copy()
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if area < min_area:
            continue
        regions.append({"x": x, "y": y, "w": w, "h": h, "area": area})
        cv2.rectangle(overlay, (x, y), (x+w, y+h), (255, 0, 0), max(2, int(a.shape[1]/700)))
    # Blend red difference mask over repro image for visibility.
    red_layer = np.zeros_like(b)
    red_layer[:, :, 0] = 255
    selected = mask > 0
    overlay[selected] = (0.55 * overlay[selected] + 0.45 * red_layer[selected]).astype(np.uint8)
    return a, b, overlay, mask, sorted(regions, key=lambda r: r["area"], reverse=True), alignment_note

def create_report(summary, images):
    out = io.BytesIO()
    c = canvas.Canvas(out, pagesize=A4)
    page_w, page_h = A4
    c.setTitle("MATCH karşılaştırma raporu")
    c.setFont("Helvetica-Bold", 18)
    c.drawString(42, page_h-48, "MATCH | PDF Karşılaştırma Raporu")
    c.setFont("Helvetica", 9)
    y = page_h-75
    for line in summary:
        c.drawString(42, y, line[:110])
        y -= 15
        if y < 100:
            c.showPage()
            y = page_h-50
            c.setFont("Helvetica", 9)
    for page_num, img in images:
        if y < 260:
            c.showPage()
            y = page_h-45
        c.setFont("Helvetica-Bold", 12)
        c.drawString(42, y, f"Sayfa {page_num}: fark haritası")
        y -= 12
        image_io = io.BytesIO()
        Image.fromarray(img).save(image_io, format="PNG")
        image_io.seek(0)
        max_w, max_h = page_w-84, y-55
        im = Image.open(image_io)
        scale = min(max_w/im.width, max_h/im.height, 1)
        c.drawImage(image_io, 42, y- im.height*scale-10, width=im.width*scale, height=im.height*scale)
        c.showPage()
        y = page_h-45
    c.save()
    out.seek(0)
    return out.getvalue()

if original_file and repro_file:
    try:
        doc_a = pdf_pages(original_file.getvalue())
        doc_b = pdf_pages(repro_file.getvalue())
        st.subheader("Dosya özeti")
        meta1, meta2, meta3 = st.columns(3)
        meta1.metric("Orijinal sayfa", len(doc_a))
        meta2.metric("Repro sayfa", len(doc_b))
        meta3.metric("Karşılaştırılacak sayfa", min(len(doc_a), len(doc_b)))
        if len(doc_a) != len(doc_b):
            st.warning("PDF'lerin sayfa sayısı farklı. Yalnızca eşleşen sayfa indeksleri karşılaştırılacak.")
        common_pages = min(len(doc_a), len(doc_b))
        if common_pages == 0:
            st.error("PDF dosyalarında karşılaştırılabilir sayfa bulunamadı.")
            st.stop()
        page_num = st.selectbox("İncelenecek sayfa", options=list(range(1, common_pages+1)), format_func=lambda x: f"Sayfa {x}")
        idx = page_num - 1
        img_a, width_a, height_a = render_page(doc_a, idx, dpi)
        img_b, width_b, height_b = render_page(doc_b, idx, dpi)
        st.caption(f"Orijinal sayfa: {width_a:.1f} × {height_a:.1f} pt | Repro sayfa: {width_b:.1f} × {height_b:.1f} pt")
        a, b, overlay, mask, regions, align_note = make_diff(img_a, img_b, threshold, min_area, align_images)
        st.caption(align_note)
        st.subheader("Görsel karşılaştırma")
        view = st.radio("Görünüm", ["Yan yana", "Bindirme / repro", "Fark haritası"], horizontal=True)
        if view == "Yan yana":
            ca, cb = st.columns(2)
            ca.image(a, caption="Müşteri orijinali", use_container_width=True)
            cb.image(b, caption="Repro", use_container_width=True)
        elif view == "Bindirme / repro":
            st.image(overlay, caption="Kırmızı bölgeler olası piksel farklarını gösterir", use_container_width=True)
        else:
            st.image(mask, caption="Beyaz alanlar eşik üstü piksel farklarıdır", use_container_width=True)
        st.subheader("Fark bölgeleri")
        st.metric("İşaretlenen bölge sayısı", len(regions))
        if regions:
            for i, r in enumerate(regions[:100], start=1):
                st.write(f"**Fark {i:02d}** — yaklaşık alan: {r['w']} × {r['h']} px; konum: x={r['x']}, y={r['y']}")
            if len(regions) > 100:
                st.caption("Ekranda ilk 100 bölge gösteriliyor.")
        else:
            st.success("Seçili eşik ve filtrelerle anlamlı fark bölgesi bulunamadı. Bu, dosyaların kesinlikle aynı olduğu anlamına gelmez.")
        if st.button("Karşılaştırma raporu oluştur", type="primary"):
            summary = [
                f"Orijinal dosya: {original_file.name}",
                f"Repro dosyası: {repro_file.name}",
                f"Sayfa: {page_num} / {common_pages}",
                f"Orijinal ölçü (pt): {width_a:.1f} x {height_a:.1f}",
                f"Repro ölçü (pt): {width_b:.1f} x {height_b:.1f}",
                f"Render DPI: {dpi}; fark eşiği: {threshold}; minimum alan: {min_area}",
                f"İşaretlenen fark bölgesi: {len(regions)}",
                "Not: Bu rapor görsel farkları işaretler; teknik baskı onayı yerine geçmez."
            ]
            pdf_bytes = create_report(summary, [(page_num, overlay)])
            st.download_button("PDF raporunu indir", data=pdf_bytes, file_name="MATCH_karsilastirma_raporu.pdf", mime="application/pdf")
        doc_a.close()
        doc_b.close()
    except Exception as e:
        st.error(f"PDF işlenirken hata oluştu: {e}")
        st.caption("Şifreli, bozuk veya desteklenmeyen PDF'ler bu prototipte açılamayabilir.")
else:
    st.markdown("### Başlamak için iki PDF yükle")
    st.write("Sol alana müşteri orijinalini, sağ alana repro dosyasını yükle.")
    st.write("İlk test için hassas olmayan örnek dosyalar kullanmanı öneririz. Dosyalar bu Streamlit uygulamasının çalıştığı ortama yüklenir; bu sürüm yerel bilgisayarında çalışmıyor.")
