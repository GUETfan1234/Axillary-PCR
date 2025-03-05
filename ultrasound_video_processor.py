#-*- coding: utf-8 -*-
import math
import os
import pydicom
from pydicom.pixel_data_handlers.util import convert_color_space, apply_voi_lut
import numpy as np
from tkinter import Tk, ttk, Canvas, Button, Scale, Label, filedialog, IntVar, StringVar, Radiobutton, Entry, Frame
from PIL import Image, ImageTk, _tkinter_finder
import matplotlib.pyplot as plt
import nibabel as nib
import cv2

class ImageCropper:
    def __init__(self, root):
        self.root = root
        self.root.title("Ultrasound Video Processor")

        self.canvas = Canvas(root, cursor="cross")
        self.canvas.pack(fill="both", expand=True)
        # 存储标注信息的列表
        self.annotations = []

        # 初始化鼠标事件变量
        self.points = []
        self.start_x = None
        self.start_y = None
        self.last_x = None
        self.last_y = None
        self.polygon = None
        self.rect = None
        self.crop_area = None
        self.drawing_mode = False  # 标注模式开关
        self.erasing_mode = False  # 擦除模式开关

        # 鼠标事件绑定
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_press)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_release)

        # # Bind mouse wheel events for Linux
        # self.canvas.bind("<Button-4>", self.on_mouse_wheel)
        # self.canvas.bind("<Button-5>", self.on_mouse_wheel)
        # Bind mouse wheel events for windows
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)

        # 顶部按钮区
        top_frame = Frame(root)
        top_frame.pack(side="top", fill="x", padx=5)

        self.select_folder_button = Button(top_frame, text="Select Folder", command=self.select_folder)
        self.select_folder_button.pack(side="left", padx=5)

        self.select_dcm_button = Button(top_frame, text="Select Video", command=self.select_video)
        self.select_dcm_button.pack(side="left", padx=5)

        self.hist_button = Button(top_frame, text="Draw Histogram", command=self.draw_histogram)
        self.hist_button.pack(side="left", padx=5)

        self.crop_button = Button(top_frame, text="Crop", command=self.set_crop_mode)
        self.crop_button.pack(side="left", padx=5)

        self.modify_button = Button(top_frame, text="Modify Pixels", command=self.set_modify_mode)
        self.modify_button.pack(side="left", padx=5)

        self.extract_button = Button(top_frame, text="Extract Range", command=self.set_extract_mode)
        self.extract_button.pack(side="left", padx=5)

        self.extract_button = Button(top_frame, text="Annotate", command=self.set_annotate_mode)
        self.extract_button.pack(side="left", padx=5)

        self.confirm_button = Button(top_frame, text="Confirm", command=self.confirm)
        self.confirm_button.pack(side="left", padx=5)


        # 底部滑块区
        bottom_frame = Frame(root)
        bottom_frame.pack(side="top", fill="x", padx=5)

        brightness_scale_label = Label(bottom_frame, text="Brightness:")
        brightness_scale_label.pack(side="left")
        self.brightness_scale = Scale(bottom_frame, from_=-255, to=255, orient="horizontal", command=self.show_frame)
        self.brightness_scale.pack(side="left", fill="x", expand=True, padx=5)

        contrast_scale_label = Label(bottom_frame, text="Contrast:")
        contrast_scale_label.pack(side="left")
        self.contrast_scale = Scale(bottom_frame, from_=-127, to=255, orient="horizontal", command=self.show_frame)
        self.contrast_scale.pack(side="left", fill="x", expand=True, padx=5)

        # 底部状态标签
        self.label = Label(root, text="")
        self.label.pack(side="bottom")

        self.image = None
        self.folder_path = None
        self.image_paths = []
        self.current_image_index = 0
        self.current_video = None
        self.current_frame_index = 0
        self.frames = []
        self.video_len = 0

        self.mode = IntVar(value=1)  # 1 for crop, 2 for modify pixels, 3 for extract range
        self.file = IntVar(value=1)   # 1 for image, 2 for video
        self.roi_var = IntVar(value=1)  # 1 for inside ROI, 2 for outside ROI

        self.in_radio = Radiobutton(bottom_frame, text="Modify Inside ROI", variable=self.roi_var, value=1)
        self.in_radio.pack(side="left", padx=5)
        self.out_radio = Radiobutton(bottom_frame, text="Modify Outside ROI", variable=self.roi_var, value=2)
        self.out_radio.pack(side="left", padx=5)

        self.pixel_value_label = Label(bottom_frame, text="Pixel Value:")
        self.pixel_value_label.pack(side="left", padx=5)
        self.pixel_value_entry = Entry(bottom_frame)
        self.pixel_value_entry.pack(side="left", padx=5)

        self.start_label = Label(bottom_frame, text="Start Image Index:")
        self.start_label.pack(side="left", padx=5)
        self.start_entry = Entry(bottom_frame)
        self.start_entry.pack(side="left", padx=5)

        self.end_label = Label(bottom_frame, text="End Image Index:")
        self.end_label.pack(side="left", padx=5)
        self.end_entry = Entry(bottom_frame)
        self.end_entry.pack(side="left", padx=5)

        # 创建标注类别选择下拉菜单
        self.annotation_type = StringVar()
        self.annotation_type.set("Label 1 Red")  # 设置默认类别
        self.annotation_menu = ttk.Combobox(bottom_frame, textvariable=self.annotation_type)
        self.annotation_menu['values'] = ("Clear Label", "Label 1 Red", "Label 2 Green", "Label 3 Blue")
        self.annotation_menu.pack(side="left", padx=5)
        # 创建透明度滑块
        self.opacity_label = Label(bottom_frame, text="Opacity:")
        self.opacity_label.pack(side="left")
        self.alpha_scale = Scale(bottom_frame, from_=0, to=100, orient="horizontal")
        self.alpha_scale.set(100)
        self.alpha_scale.pack(side="left", padx=5)
        self.alpha_scale.bind("<Motion>", self.update_opacity)
        # 创建按钮
        # self.annotate_button = Button(bottom_frame, text="Draw", command=self.toggle_annotation_mode)
        # self.annotate_button.pack(side="left", padx=5)
        # self.erase_button = Button(bottom_frame, text="Erase", command=self.toggle_erase_mode)
        # self.erase_button.pack(side="left", padx=5)
        self.clear_button = Button(bottom_frame, text="Clear", command=self.clear_annotations)
        self.clear_button.pack(side="left", padx=5)

        self.drag_data = {"x": 0, "y": 0, "item": None}
        self.resizing = None
        self.dragging = False
        self.canvas.bind("<Motion>", self.on_motion)

        self.in_radio.pack_forget()
        self.out_radio.pack_forget()
        self.pixel_value_label.pack_forget()
        self.pixel_value_entry.pack_forget()
        self.start_label.pack_forget()
        self.start_entry.pack_forget()
        self.end_label.pack_forget()
        self.end_entry.pack_forget()
        self.annotation_menu.pack_forget()
        self.opacity_label.pack_forget()
        self.alpha_scale.pack_forget()
        # self.annotate_button.pack_forget()
        # self.erase_button.pack_forget()
        self.clear_button.pack_forget()

    def set_crop_mode(self):
        if self.file.get() == 1:
            self.mode.set(1)
        elif self.file.get() == 2:
            self.mode.set(4)
        self.in_radio.pack_forget()
        self.out_radio.pack_forget()
        self.pixel_value_label.pack_forget()
        self.pixel_value_entry.pack_forget()
        self.start_label.pack_forget()
        self.start_entry.pack_forget()
        self.end_label.pack_forget()
        self.end_entry.pack_forget()
        self.annotation_menu.pack_forget()
        self.opacity_label.pack_forget()
        self.alpha_scale.pack_forget()
        # self.annotate_button.pack_forget()
        # self.erase_button.pack_forget()
        self.clear_button.pack_forget()
        self.label.config(text="Crop mode selected")

    def set_modify_mode(self):
        if self.file.get() == 1:
            self.mode.set(2)
        elif self.file.get() == 2:
            self.mode.set(5)
        self.in_radio.pack(side="left", padx=5)
        self.out_radio.pack(side="left", padx=5)
        self.pixel_value_label.pack(side="left", padx=5)
        self.pixel_value_entry.pack(side="left", padx=5)
        self.start_label.pack_forget()
        self.start_entry.pack_forget()
        self.end_label.pack_forget()
        self.end_entry.pack_forget()
        self.annotation_menu.pack_forget()
        self.opacity_label.pack_forget()
        self.alpha_scale.pack_forget()
        # self.annotate_button.pack_forget()
        # self.erase_button.pack_forget()
        self.clear_button.pack_forget()
        self.label.config(text="Modify Pixels mode selected")

    def set_extract_mode(self):
        if self.file.get() == 1:
            self.mode.set(3)
        elif self.file.get() == 2:
            self.mode.set(6)
        self.in_radio.pack_forget()
        self.out_radio.pack_forget()
        self.pixel_value_label.pack_forget()
        self.pixel_value_entry.pack_forget()
        self.start_label.pack(side="left", padx=5)
        self.start_entry.pack(side="left", padx=5)
        self.end_label.pack(side="left", padx=5)
        self.end_entry.pack(side="left", padx=5)
        self.annotation_menu.pack_forget()
        self.opacity_label.pack_forget()
        self.alpha_scale.pack_forget()
        # self.annotate_button.pack_forget()
        # self.erase_button.pack_forget()
        self.clear_button.pack_forget()
        self.label.config(text="Extract Range mode selected")

    def set_annotate_mode(self):
        if self.file.get() == 1:
            self.mode.set(7)
        elif self.file.get() == 2:
            self.mode.set(8)
        self.in_radio.pack_forget()
        self.out_radio.pack_forget()
        self.pixel_value_label.pack_forget()
        self.pixel_value_entry.pack_forget()
        self.start_label.pack_forget()
        self.start_entry.pack_forget()
        self.end_label.pack_forget()
        self.end_entry.pack_forget()
        self.annotation_menu.pack(side="left", padx=5)
        self.opacity_label.pack(side="left", padx=5)
        self.alpha_scale.pack(side="left", padx=5)
        # self.annotate_button.pack(side="left", padx=5)
        # self.erase_button.pack(side="left", padx=5)
        self.clear_button.pack(side="left", padx=5)
        self.label.config(text="Annotate mode selected")

    def select_folder(self):
        self.label.config(text=f"selecting directory...")
        self.folder_path = filedialog.askdirectory()
        self.file.set(1)
        if self.folder_path:
            self.image_paths = self.get_image_paths(self.folder_path)
            if self.image_paths:
                self.current_image_index = 0
                self.open_image(self.image_paths[self.current_image_index])
                self.label.config(text=f"directory selected!")
        else:
            self.label.config(text=f"failed open {self.folder_path}!")

    def select_video(self):
        self.label.config(text=f"selecting video...")
        file_path = filedialog.askopenfilename(filetypes=[("Video files", ["*.dcm", "*.DCM", "*.MP4", "*.mp4",
                                                                           "*.avi", "*.AVI", '.WMV', '.wmv',
                                                                           '*.nii', '*.nii.gz'])])

        self.file.set(2)
        # print(file_path)
        if file_path:
            self.current_video = file_path
            self.folder_path = file_path
            self.open_video(file_path)
            self.label.config(text=f"video selected!")
        else:
            self.label.config(text=f"failed open {file_path}!")

    def get_image_paths(self, folder):
        image_paths = []
        # print(folder)
        for root, _, files in os.walk(folder):
            sorted_files = sorted(os.path.join(root, file) for file in files)
            for file in sorted_files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    image_paths.append(os.path.join(root, file))
        return image_paths

    def open_image(self, image_path):
        if image_path.lower().endswith(('.dcm', '.avi', '.mp4', '.wmv', '*.nii', '*.nii.gz')):
            self.open_video(image_path)
        else:
            self.image = Image.open(image_path)
            self.image_tk = ImageTk.PhotoImage(self.image)
            self.canvas.config(width=self.image_tk.width(), height=self.image_tk.height())
            self.canvas.create_image(0, 0, anchor="nw", image=self.image_tk)
            self.label.config(text=f"Displaying {os.path.basename(image_path)}, index:{self.image_paths.index(image_path)}")
        if self.crop_area:
            self.rect = self.canvas.create_rectangle(self.crop_area, outline="red", width=2)

    def open_video(self, video_path):
        if video_path.lower().endswith(".dcm"):
            ds = pydicom.dcmread(video_path)
            self.mask = np.zeros(ds.pixel_array.shape[:3], dtype=np.uint8)
            # print(ds.pixel_array.shape)
            if 'NumberOfFrames' in ds:
                if ds.PhotometricInterpretation == 'YBR_FULL_422':
                    self.frames = [convert_color_space(ds.pixel_array[i], 'YBR_FULL_422', 'RGB') for i in range(ds.NumberOfFrames)]
                elif ds.PhotometricInterpretation == 'RGB':
                    self.frames = [ds.pixel_array[i] for i in range(ds.NumberOfFrames)]
                self.video_len = len(self.frames)
                self.current_frame_index = 0
                self.show_frame()
            else:
                img_array = ds.pixel_array
                img = Image.fromarray(img_array)
                self.image = img.convert("L")  # convert to grayscale if needed
                self.image_tk = ImageTk.PhotoImage(self.image)
                self.canvas.config(width=self.image_tk.width(), height=self.image_tk.height())
                self.canvas.create_image(0, 0, anchor="nw", image=self.image_tk)
                self.label.config(text=f"Displaying {os.path.basename(video_path)}")
        elif video_path.lower().endswith((".avi", '.mp4', '.wmv')):
            self.video_capture = cv2.VideoCapture(video_path)
            width = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.video_len = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            self.mask = np.zeros((self.video_len, height, width), dtype=np.uint8)
            self.current_frame_index = 0
            if self.video_capture.isOpened():
                self.show_frame()
        elif video_path.lower().endswith(('.nii', '.nii.gz')):
            self.nii_data = nib.load(video_path).get_fdata()
            self.mask = np.zeros(self.nii_data.shape[:3], dtype=np.uint8)
            # print(self.nii_data.shape)
            if len(self.nii_data.shape) == 5 and self.nii_data.shape[3] == 1:
                self.nii_data = self.nii_data[:, :, :, 0, :]
            # elif len(self.nii_data.shape) == 3:
            #     self.nii_data = np.repeat(self.nii_data[:, :, :, np.newaxis], 3, axis=2)
            self.video_len = self.nii_data.shape[2]
            self.current_frame_index = 0
            self.frames = [np.uint8(self.nii_data[:, :, i, ...]) for i in range(self.nii_data.shape[2])]
            self.show_frame()

    def show_frame(self, event=None):
        frame_index = self.current_frame_index
        if self.current_video.lower().endswith(".dcm"):
            self.current_frame = np.array(self.frames[frame_index])
            self.original_frame = self.current_frame.copy()
            if self.crop_area:
                img = self.current_frame
                cropped_img_array = img[self.crop_area[1]:self.crop_area[3], self.crop_area[0]:self.crop_area[2], :]
                adjusted_img = self.adjust_brightness_contrast(cropped_img_array, self.brightness_scale.get(),
                                                                 self.contrast_scale.get())
                img[self.crop_area[1]:self.crop_area[3], self.crop_area[0]:self.crop_area[2], :] = adjusted_img
                adjusted_frame = img
            else:
                adjusted_frame = self.adjust_brightness_contrast(self.current_frame, self.brightness_scale.get(),
                                                                 self.contrast_scale.get())
            image = Image.fromarray(adjusted_frame)
            self.image_tk = ImageTk.PhotoImage(image)
            self.canvas.config(width=self.image_tk.width(), height=self.image_tk.height())
            self.canvas.create_image(0, 0, anchor="nw", image=self.image_tk)
            self.label.config(text=f"Displaying frame {frame_index + 1}/{self.video_len} of {os.path.basename(self.current_video)}")
        elif self.current_video.lower().endswith((".avi", '.mp4', '.wmv')):
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_index)
            ret, self.current_frame = self.video_capture.read()
            self.original_frame = self.current_frame.copy()
            if ret:
                # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if self.crop_area:
                    img = self.current_frame
                    cropped_img_array = img[self.crop_area[1]:self.crop_area[3], self.crop_area[0]:self.crop_area[2], :]
                    adjusted_img = self.adjust_brightness_contrast(cropped_img_array, self.brightness_scale.get(),
                                                                   self.contrast_scale.get())
                    img[self.crop_area[1]:self.crop_area[3], self.crop_area[0]:self.crop_area[2], :] = adjusted_img
                    adjusted_frame = img
                else:
                    adjusted_frame = self.adjust_brightness_contrast(self.current_frame, self.brightness_scale.get(),
                                                                     self.contrast_scale.get())
                image = Image.fromarray(adjusted_frame)
                self.image_tk = ImageTk.PhotoImage(image)
                self.canvas.config(width=self.image_tk.width(), height=self.image_tk.height())
                self.canvas.create_image(0, 0, anchor="nw", image=self.image_tk)
                self.label.config(
                    text=f"Displaying frame {self.current_frame_index + 1}/{self.video_len} of {os.path.basename(self.current_video)}")
        elif self.current_video.lower().endswith(('.nii', '.nii.gz')):
            image = np.array(self.frames[frame_index])
            self.current_frame = np.array(image)
            self.original_frame = self.current_frame.copy()
            if len(image.shape) == 3:
                image = np.transpose(image, [1, 0, 2])
            elif len(image.shape) == 2:
                image = np.transpose(image)
                image = np.flip(image, 0)
                image = np.flip(image, 1)
            # image = np.flip(image, 1)
            # image = np.flip(image, 0)
            adjusted_frame = self.adjust_brightness_contrast(image, self.brightness_scale.get(),
                                                             self.contrast_scale.get())
            self.image = Image.fromarray(adjusted_frame)
            self.image_tk = ImageTk.PhotoImage(self.image)
            self.canvas.config(width=self.image_tk.width(), height=self.image_tk.height())
            self.canvas.create_image(0, 0, anchor="nw", image=self.image_tk)
            self.label.config(
                text=f"Displaying frame {self.current_frame_index + 1}/{self.video_len} of {os.path.basename(self.current_video)}")
        if self.crop_area:
            self.rect = self.canvas.create_rectangle(self.crop_area, outline="red", width=2)
            # self.canvas.coords(self.rect, *self.crop_area)

    def adjust_brightness_contrast(self, image, brightness=0, contrast=0):
        beta = brightness
        alpha = contrast / 127.0 + 1.0
        return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    def draw_histogram(self):
        # 初始化直方图数组
        hist = np.zeros((256,), dtype=np.float32)
        if self.current_video.lower().endswith(".dcm"):
            for frame in self.frames:
                if self.crop_area:
                    img = np.array(frame)
                    cropped_img_array = img[self.crop_area[1]:self.crop_area[3], self.crop_area[0]:self.crop_area[2], :]
                    adjusted_frame = self.adjust_brightness_contrast(cropped_img_array, self.brightness_scale.get(),
                                                                   self.contrast_scale.get())
                else:
                    adjusted_frame = self.adjust_brightness_contrast(frame, self.brightness_scale.get(),
                                                                     self.contrast_scale.get())
                gray_frame = cv2.cvtColor(adjusted_frame, cv2.COLOR_RGB2GRAY)
                # 计算当前帧的直方图并累加
                frame_hist = cv2.calcHist([gray_frame], [0], None, [256], [0, 256])
                hist += frame_hist.flatten()
        elif self.current_video.lower().endswith((".avi", '.mp4', '.wmv')):
            # 重置视频到起始位置
            cap = self.video_capture
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                # 将图像转换为灰度图
                if self.crop_area:
                    img = np.array(frame)
                    cropped_img_array = img[self.crop_area[1]:self.crop_area[3], self.crop_area[0]:self.crop_area[2], :]
                    adjusted_img = self.adjust_brightness_contrast(cropped_img_array, self.brightness_scale.get(),
                                                                   self.contrast_scale.get())
                    img[self.crop_area[1]:self.crop_area[3], self.crop_area[0]:self.crop_area[2], :] = adjusted_img
                    adjusted_frame = img
                else:
                    adjusted_frame = self.adjust_brightness_contrast(frame, self.brightness_scale.get(),
                                                                     self.contrast_scale.get())
                gray_frame = cv2.cvtColor(adjusted_frame, cv2.COLOR_RGB2GRAY)

                # 计算当前帧的直方图并累加
                frame_hist = cv2.calcHist([gray_frame], [0], None, [256], [0, 256])
                hist += frame_hist.flatten()

        hist = (hist / hist.sum()) * 100
        # 绘制累积直方图
        plt.figure()
        plt.title("Video Gray Scale Histogram")
        plt.xlabel("Pixel Value")
        plt.ylabel("Percent of Pixels")
        # plt.plot(hist)
        plt.bar(range(256), hist, width=2)
        plt.xlim([-2, 258])
        plt.show()

    # def on_mouse_wheel(self, event):
    #     if self.file.get() == 2:
    #         if event.num == 4:
    #             self.current_frame_index = (self.current_frame_index - 1) % self.video_len
    #         elif event.num == 5:
    #             self.current_frame_index = (self.current_frame_index + 1) % self.video_len
    #         self.show_frame()
    #     elif self.file.get() == 1:
    #         if event.num == 4:
    #             self.current_image_index = (self.current_image_index - 1) % len(self.image_paths)
    #         elif event.num == 5:
    #             self.current_image_index = (self.current_image_index + 1) % len(self.image_paths)
    #         self.open_image(self.image_paths[self.current_image_index])


    def on_mouse_wheel(self, event):
        if self.file.get() == 2:
            if event.delta > 0:
                self.current_frame_index = (self.current_frame_index - 1) % self.video_len
            elif event.delta < 0:
                self.current_frame_index = (self.current_frame_index + 1) % self.video_len
            self.show_frame(self.current_frame_index)
        elif self.file.get() == 1:
            if event.delta > 0:
                self.current_image_index = (self.current_image_index - 1) % len(self.image_paths)
            elif event.delta < 0:
                self.current_image_index = (self.current_image_index + 1) % len(self.image_paths)
            self.open_image(self.image_paths[self.current_image_index])

    # def on_button_press(self, event):
    #     self.start_x = event.x
    #     self.start_y = event.y
    #     if self.rect:
    #         self.canvas.delete(self.rect)
    #     self.rect = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline="red")

    def on_mouse_press(self, event):
        # 鼠标按下时，记录起点位置
        if self.mode.get() == 7 or self.mode.get() == 8:
            self.start_x, self.start_y = event.x, event.y
            self.last_x, self.last_y = self.start_x, self.start_y
            self.points = [(self.start_x, self.start_y)]
        else:
            # Check if click is on the resize handle
            if self.rect and self.canvas.coords(self.rect):
                x1, y1, x2, y2 = self.canvas.coords(self.rect)
                if x1 - 10 <= event.x <= x1 + 10 and y1 - 10 <= event.y <= y1 + 10:
                    self.resizing = 1
                    return
                elif x1 + 10 <= event.x <= x2 - 10 and y1 - 10 <= event.y <= y1 + 10:
                    self.resizing = 2
                    return
                elif x2 - 10 <= event.x <= x2 + 10 and y1 - 10 <= event.y <= y1 + 10:
                    self.resizing = 3
                    return
                elif x2 - 10 <= event.x <= x2 + 10 and y1 + 10 <= event.y <= y2 - 10:
                    self.resizing = 4
                    return
                elif x2 - 10 <= event.x <= x2 + 10 and y2 - 10 <= event.y <= y2 + 10:
                    self.resizing = 5
                    return
                elif x1 + 10 <= event.x <= x2 - 10 and y2 - 10 <= event.y <= y2 + 10:
                    self.resizing = 6
                    return
                elif x1 - 10 <= event.x <= x1 + 10 and y2 - 10 <= event.y <= y2 + 10:
                    self.resizing = 7
                    return
                elif x1 - 10 <= event.x <= x1 + 10 and y1 + 10 <= event.y <= y2 - 10:
                    self.resizing = 8
                    return
                else:
                    self.resizing = 0
                # print(self.resizing)
            # Otherwise start a new rectangle
                self.canvas.delete(self.rect)
            self.crop_area = [event.x, event.y, event.x, event.y]
            self.rect = self.canvas.create_rectangle(self.crop_area, outline="red", width=2)
            self.drag_data = {"x": event.x, "y": event.y, "item": self.rect}

    def on_mouse_drag(self, event):
        if self.mode.get() == 7 or self.mode.get() == 8:
            # 鼠标移动时，计算当前点和上一个点之间的距离
            cur_x, cur_y = event.x, event.y
            distance = math.sqrt((cur_x - self.last_x) ** 2 + (cur_y - self.last_y) ** 2)

            # 如果移动距离超过一定阈值，绘制线段并记录点
            if distance > 5:  # 距离阈值，可以根据需求调整
                self.canvas.create_line(self.last_x, self.last_y, cur_x, cur_y, fill='white')
                self.points.append((cur_x, cur_y))
                self.last_x, self.last_y = cur_x, cur_y
        else:
            if self.resizing:
                # Resize the rectangle
                if self.resizing == 1:
                    self.crop_area[0] = event.x
                    self.crop_area[1] = event.y
                elif self.resizing == 2:
                    self.crop_area[1] = event.y
                elif self.resizing == 3:
                    self.crop_area[2] = event.x
                    self.crop_area[1] = event.y
                elif self.resizing == 4:
                    self.crop_area[2] = event.x
                elif self.resizing == 5:
                    self.crop_area[2] = event.x
                    self.crop_area[3] = event.y
                elif self.resizing == 6:
                    self.crop_area[3] = event.y
                elif self.resizing == 7:
                    self.crop_area[0] = event.x
                    self.crop_area[3] = event.y
                elif self.resizing == 8:
                    self.crop_area[0] = event.x
                self.canvas.coords(self.rect, *self.crop_area)
            elif self.drag_data["item"]:
                self.crop_area[2] = event.x
                self.crop_area[3] = event.y
                self.canvas.coords(self.rect, *self.crop_area)

    def on_mouse_release(self, event):
        if self.mode.get() == 7 or self.mode.get() == 8:
            # 鼠标松开时，判断是否在起点附近
            cur_x, cur_y = self.start_x, self.start_y

            self.canvas.create_line(self.last_x, self.last_y, cur_x, cur_y, fill='red')
            self.points.append((cur_x, cur_y))

            self.polygon = self.points
            if self.annotation_type.get() == "Clear Label":
                self.erase_polygon()
            else:
                self.fill_polygon()

            # 清空顶点列表
            self.points = []
        else:
            self.drag_data = {"x": 0, "y": 0, "item": None}
            self.resizing = False
            # self.crop_area[2] = event.x
            # self.crop_area[3] = event.y
            self.canvas.coords(self.rect, *self.crop_area)
            self.label.config(text=f"Crop area: {self.crop_area}")

    def fill_polygon(self):
        # 获取多边形区域
        if self.polygon:
            poly = np.array(self.polygon)
            mask = self.create_polygon_mask(self.current_frame.shape[:2], poly)
            # 将多边形区域标记为所选类别颜色，应用透明度
            if self.annotation_type.get() == "Label 1 Red":
                label = 1
                color = [255, 0, 0]  # 红色
            elif self.annotation_type.get() == "Label 2 Green":
                label = 2
                color = [0, 255, 0]  # 绿色
            elif self.annotation_type.get() == "Label 3 Blue":
                label = 3
                color = [0, 0, 255]  # 蓝色

            self.mask[self.current_frame_index][mask] = label
            alpha = self.alpha_scale.get() / 100.0  # 获取透明度
            # self.annotations.append((mask, color))  # 保存标注
            self.apply_all_annotations()
            # self.apply_mask(mask, color, alpha)

            # # print(mask.shape)
            # # mask_rgb = np.zeros_like(self.current_frame)
            # # mask_rgb[mask] = [255, 0, 0]
            # # 将多边形区域标记为红色
            # # self.current_frame = np.where(mask_rgb != 0, mask_rgb, self.current_frame)
            # self.current_frame[mask] = [255, 0, 0]
            #
            # self.refresh_canvas()

    def erase_polygon(self):
        # 获取多边形区域
        if self.polygon:
            poly = np.array(self.polygon)
            mask = self.create_polygon_mask(self.current_frame.shape[:2], poly)
            # 移除标注列表中与擦除区域重叠的标注
            self.mask[self.current_frame_index][mask] = 0
            # 恢复对应区域的原始帧颜色
            self.current_frame[mask] = self.original_frame[mask]
            self.refresh_canvas()

    def create_polygon_mask(self, shape, polygon):
        # 创建一个与图像同样大小的空白掩膜
        mask = np.zeros(shape, dtype=np.uint8)
        # 使用 OpenCV 填充多边形
        cv2.fillPoly(mask, [polygon], 1)

        return mask.astype(bool)

    def apply_mask(self, mask, color, alpha=None):
        if color is not None:  # 绘制标注
            overlay = np.zeros_like(self.current_frame)
            overlay[mask] = color

            # 混合原图像和标注
            overlay = cv2.addWeighted(self.current_frame, 1.0-alpha, overlay, alpha, 1)
            self.current_frame[mask] = overlay[mask]
        else:  # 擦除标注
            self.current_frame[mask] = self.original_frame[mask]

        self.refresh_canvas()

    def apply_all_annotations(self):
        # 重新应用所有标注
        self.current_frame = self.original_frame.copy()
        alpha = self.alpha_scale.get() / 100.0  # 获取透明度
        # for mask, color in self.annotations:
        #     if color is None:
        #         self.current_frame[mask] = self.original_frame[mask]
        #         continue
        #     overlay = np.zeros_like(self.current_frame)
        #     overlay[mask] = color
        #     overlay = cv2.addWeighted(self.current_frame, 1.0 - alpha, overlay, alpha, 1)
        #     self.current_frame[mask] = overlay[mask]
        # 根据mask填充颜色
        overlay = np.zeros_like(self.current_frame)
        overlay[self.mask[self.current_frame_index] == 1] = [255, 0, 0]  # 类别A颜色
        overlay[self.mask[self.current_frame_index] == 2] = [0, 255, 0]  # 类别B颜色
        overlay[self.mask[self.current_frame_index] == 3] = [0, 0, 255]  # 类别C颜色
        overlay = cv2.addWeighted(self.current_frame, 1.0 - alpha, overlay, alpha, 1)
        self.current_frame[self.mask[self.current_frame_index] == 1] = overlay[self.mask[self.current_frame_index] == 1]
        self.current_frame[self.mask[self.current_frame_index] == 2] = overlay[self.mask[self.current_frame_index] == 2]
        self.current_frame[self.mask[self.current_frame_index] == 3] = overlay[self.mask[self.current_frame_index] == 3]
        self.refresh_canvas()

    def update_opacity(self, event=None):
        # 在滑动透明度条时，更新所有标注的透明度
        self.apply_all_annotations()

    def clear_annotations(self):
        # 清除标注并恢复原始图像
        self.mask[self.current_frame_index].fill(0)
        self.current_frame = self.original_frame.copy()
        self.refresh_canvas()

    def refresh_canvas(self):
        # 更新Canvas显示
        self.image_tk = ImageTk.PhotoImage(image=Image.fromarray(self.current_frame))
        self.canvas.config(width=self.image_tk.width(), height=self.image_tk.height())
        self.canvas.create_image(0, 0, anchor="nw", image=self.image_tk)
        # self.photo = ImageTk.PhotoImage(image=Image.fromarray(self.current_frame))
        # # self.canvas.itemconfig(self.current_frame, image=self.photo)
        # self.canvas.image = self.photo  # 避免图像被垃圾回收

    def save_annotation(self):
        output_path = os.path.dirname(self.current_video) + '_annotate/' + os.path.basename(self.current_video)
        # print(output_folder_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        img_array = np.array(self.mask)
        if self.crop_area:
            img_array = img_array[:, self.crop_area[1]:self.crop_area[3], self.crop_area[0]:self.crop_area[2]]
        output_nii_path = output_path.replace(os.path.splitext(output_path)[1], '.nii.gz')
        nib.save(nib.Nifti1Image(np.fliplr(np.rot90(np.transpose(img_array, [1, 2, 0]))), np.eye(4)), output_nii_path)
        # print(output_dcm_path)
        self.label.config(text=f"Annotation saved as {output_nii_path}")


    # def on_button_press(self, event):
    #     # Check if click is on the resize handle
    #     if self.rect and self.canvas.coords(self.rect):
    #         x1, y1, x2, y2 = self.canvas.coords(self.rect)
    #         if x1 - 10 <= event.x <= x1 + 10 and y1 - 10 <= event.y <= y1 + 10:
    #             self.resizing = 1
    #             return
    #         elif x1 + 10 <= event.x <= x2 - 10 and y1 - 10 <= event.y <= y1 + 10:
    #             self.resizing = 2
    #             return
    #         elif x2 - 10 <= event.x <= x2 + 10 and y1 - 10 <= event.y <= y1 + 10:
    #             self.resizing = 3
    #             return
    #         elif x2 - 10 <= event.x <= x2 + 10 and y1 + 10 <= event.y <= y2 - 10:
    #             self.resizing = 4
    #             return
    #         elif x2 - 10 <= event.x <= x2 + 10 and y2 - 10 <= event.y <= y2 + 10:
    #             self.resizing = 5
    #             return
    #         elif x1 + 10 <= event.x <= x2 - 10 and y2 - 10 <= event.y <= y2 + 10:
    #             self.resizing = 6
    #             return
    #         elif x1 - 10 <= event.x <= x1 + 10 and y2 - 10 <= event.y <= y2 + 10:
    #             self.resizing = 7
    #             return
    #         elif x1 - 10 <= event.x <= x1 + 10 and y1 + 10 <= event.y <= y2 - 10:
    #             self.resizing = 8
    #             return
    #         else:
    #             self.resizing = 0
    #         # print(self.resizing)
    #     # Otherwise start a new rectangle
    #         self.canvas.delete(self.rect)
    #     self.crop_area = [event.x, event.y, event.x, event.y]
    #     self.rect = self.canvas.create_rectangle(self.crop_area, outline="red", width=2)
    #     self.drag_data = {"x": event.x, "y": event.y, "item": self.rect}
    #
    # # def on_mouse_drag(self, event):
    # #     self.canvas.coords(self.rect, self.start_x, self.start_y, event.x, event.y)
    #
    # def on_mouse_drag(self, event):
    #     if self.resizing:
    #         # Resize the rectangle
    #         if self.resizing == 1:
    #             self.crop_area[0] = event.x
    #             self.crop_area[1] = event.y
    #         elif self.resizing == 2:
    #             self.crop_area[1] = event.y
    #         elif self.resizing == 3:
    #             self.crop_area[2] = event.x
    #             self.crop_area[1] = event.y
    #         elif self.resizing == 4:
    #             self.crop_area[2] = event.x
    #         elif self.resizing == 5:
    #             self.crop_area[2] = event.x
    #             self.crop_area[3] = event.y
    #         elif self.resizing == 6:
    #             self.crop_area[3] = event.y
    #         elif self.resizing == 7:
    #             self.crop_area[0] = event.x
    #             self.crop_area[3] = event.y
    #         elif self.resizing == 8:
    #             self.crop_area[0] = event.x
    #         self.canvas.coords(self.rect, *self.crop_area)
    #     elif self.drag_data["item"]:
    #         self.crop_area[2] = event.x
    #         self.crop_area[3] = event.y
    #         self.canvas.coords(self.rect, *self.crop_area)
    #
    # # def on_button_release(self, event):
    # #     self.crop_area = (self.start_x, self.start_y, event.x, event.y)
    # #     self.label.config(text=f"Crop area: {self.crop_area}")
    #
    # def on_button_release(self, event):
    #     self.drag_data = {"x": 0, "y": 0, "item": None}
    #     self.resizing = False
    #     # self.crop_area[2] = event.x
    #     # self.crop_area[3] = event.y
    #     self.canvas.coords(self.rect, *self.crop_area)
    #     self.label.config(text=f"Crop area: {self.crop_area}")

    def on_motion(self, event):
        if self.resizing or self.dragging:
            return

        if self.rect:
            x1, y1, x2, y2 = self.canvas.coords(self.rect)
            if x1 - 10 <= event.x <= x2 + 10 and y1 - 10 <= event.y <= y1 + 10 or\
               x1 - 10 <= event.x <= x2 + 10 and y2 - 10 <= event.y <= y2 + 10 or\
               x1 - 10 <= event.x <= x1 + 10 and y1 - 10 <= event.y <= y2 + 10 or\
               x2 - 10 <= event.x <= x2 + 10 and y1 - 10 <= event.y <= y2 + 10:
                self.canvas.config(cursor="fleur")
            else:
                self.canvas.config(cursor="")
        else:
            self.canvas.config(cursor="")

    def confirm(self):
        if self.mode.get() == 1:
            self.crop_images()
        elif self.mode.get() == 2:
            self.modify_pixels()
        elif self.mode.get() == 3:
            self.extract_images()
        elif self.mode.get() == 4:
            self.crop_video()
        elif self.mode.get() == 5:
            self.modify_video_pixels()
        elif self.mode.get() == 6:
            self.extract_video_frames()
        elif self.mode.get() == 8:
            self.save_annotation()

if __name__ == "__main__":
    root = Tk()
    app = ImageCropper(root)
    root.mainloop()