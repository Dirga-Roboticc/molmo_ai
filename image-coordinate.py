# Memuat gambar dan menggambar titik sesuai instruksi user
from PIL import Image, ImageDraw

# Memuat gambar yang sudah ada
image_path = './output/Screenshot_20240920_142800.png'
img = Image.open(image_path)

# Ukuran gambar (width, height)
width, height = img.size

# Koordinat yang diberikan sebagai rasio (x: 5.7, y: 11.7)
# Mengonversi ke piksel sebenarnya berdasarkan ukuran gambar
x = int(7.0 / 100 * width)
y = int(22.9 / 100 * height)

# Membuat objek untuk menggambar
draw = ImageDraw.Draw(img)

# Menandai titik dengan warna biru muda (light blue)
radius = 10
draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill="lightblue", outline="lightblue")

# Tampilkan gambar dengan titik
img.save('output.png')