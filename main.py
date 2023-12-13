from ultralytics import YOLO
from modules.jersey_module import klasifikasi_warna_jersey
from modules.hardwarecheck import cudagpu

import cv2
import os
import time

cudagpu()

# Load model
model = YOLO('weights/football-scouting-best-m-1695-aug-segonlyplayer.pt')
# results = model('sample_2.jpg', show=True, save=True) # Contoh prediksi pada single image

# Membaca file video.mp4
VIDEO_PATH = "input/ec9f4e2b_1.mp4" # Source video dalam lokal
cap = cv2.VideoCapture(VIDEO_PATH) # 0 = webcam, 1 = external webcam, VIDEO_PATH = lokasi video lokal

# Membaca detail frame video
file_base = os.path.basename(VIDEO_PATH) # Get nama direktori file dan nama file
file_name = os.path.splitext(file_base) # Get nama file saja 
frame_jumlah = cap.get(cv2.CAP_PROP_FRAME_COUNT) # Get total frame dalam file video masukan
fps = cap.get(cv2.CAP_PROP_FPS)  # Get FPS (Frame Per Second) dalam file video masukan
frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH) # Get ukuran width dari frame video masukan
frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) # Get ukuran height dari frame video masukan

# Menampilkan informasi video
print("============= DETAIL FILE VIDEO =============")
print(f"Nama File Video :", file_name, 
      "\nTotal Frame :", int(frame_jumlah), 
      "\nFPS :", int(fps), 
      "\nDurasi Video (Detik) :", frame_jumlah/fps,
      "\nOriginal Ukuran Frame :", int(frame_width), int(frame_height),
      "\nModel Label/Kelas :", model.names),
print("=============================================")

# Simpan video ke ukuran 1280x720 (supaya lebih efisien & kompresi ukuran file)
NEW_FRAME_WIDTH = 1280
NEW_FRAME_HEIGHT = 720
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Format video
OUTPUT_PATH_FOLDER = os.path.join("output/videos\playerball", "football-scouting-best-m-1695-aug-segonlyplayer.pt") # Lokasi simpan video hasil inference
try:
  os.makedirs(OUTPUT_PATH_FOLDER)
  print("Folder %s terbuat!\n" % OUTPUT_PATH_FOLDER)
except FileExistsError:
  print("Folder %s telah tersedia\n" % OUTPUT_PATH_FOLDER)
OUTPUT_PATH_VIDEOS = OUTPUT_PATH_FOLDER+f"/{file_name[0]} output.mp4"
out = cv2.VideoWriter(OUTPUT_PATH_VIDEOS, fourcc, fps, (NEW_FRAME_WIDTH, NEW_FRAME_HEIGHT)) # Untuk simpan

#################
frame_nomor = 0
#################

while cap.isOpened(): # True
    start = time.time()
    success, frame = cap.read()     # Membaca frame saat ini yang telah diekstrak
    frame_nomor += 1

    if success:
        # Jalankan inference/prediksi pada frame saat ini dan persisting tracks between frames 
        results = model(frame, imgsz=1280, conf=0.6)
        print(f"Memproses Frame Urutan ke", frame_nomor)

        # Menampilkan hasil prediksi (per bunding box) pada frame ini
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # print(box)
                # print(box.cls[0])
                kelas = int(box.cls[0])
                # print(kelas)
                # print(box.xyxy[0])
                # print(x1,y1,x2,y2)
                # print(x1,y1,x2,y2)
                x1, y1, x2, y2 = box.xyxy[0] # Mencari titik x1,y1 dan x2,y2
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                if kelas == 2: # Jika kelas ini player
                    warnajersey = klasifikasi_warna_jersey(frame[y1:y2, x1:x2])
                    bbox_frame = cv2.rectangle(frame, (x1,y1), (x2,y2), warnajersey, 2)
                elif kelas == 0: # Jika kelas ini bola
                    bbox_frame = cv2.rectangle(frame, (x1,y1), (x2,y2), (255,255,255), 2)
                else:
                    bbox_frame = frame

        # Resize dari 1920x1080 ke 1280x720
        resized_frame = cv2.resize(bbox_frame, (NEW_FRAME_WIDTH, NEW_FRAME_HEIGHT))

        end = time.time()
        fps_count = 1/(end-start)
        cv2.putText(resized_frame, f"FPS : {int(fps)}", (30,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Simpan/add per frame ke format video
        out.write(resized_frame)

        # # Visualize the results on the frame
        # annotated_frame = results[0].plot()

        # Resize dari 1920x1080 ke 1280x720
        resized_frame = cv2.resize(resized_frame, (NEW_FRAME_WIDTH, NEW_FRAME_HEIGHT))

        # Tampilkan hasil di layar
        cv2.imshow("{} Tracking".format(file_name[0]), resized_frame)
        # Untuk menghentikan looping ekstraksi frame dari video dengan menekan 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Berhenti ketika sampai frame terakhir
        break

# Release the video capture object and close the display window
print("\nOutput video telah berhasil disimpan pada '{}!'".format(OUTPUT_PATH_VIDEOS))
cap.release()
out.release()
cv2.destroyAllWindows()