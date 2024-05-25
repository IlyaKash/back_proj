import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import Scale, IntVar, Radiobutton
from PIL import ImageTk, Image

class ImageProcessor:
    def __init__(self, root, folder_path, masks_path, saving_path):
        self.root = root
        self.frame_menu = tk.Frame(borderwidth=1, relief="solid")
        self.frame_images = tk.Frame(borderwidth=1, relief="solid")  # Новый фрейм для сетки изображений
        self.folder_path = masks_path
        self.orig_img_path= folder_path
        self.saving_path=saving_path
        self.image_files = self.load_images()
        self.orig_img_files=self.load_orig_img()
        self.current_image_index = 0
        self.processed_images = [None] * 9
        self.processed_orig_images = [None] * 9
        self.images_for_saving=[None]*9
        self.enabled=tk.BooleanVar()
        self.enabled.set(False)


        self.kernel_size_dilation = 1
        self.kernel_size_erosion = 1
        self.count_of_iterations_dilation = 1
        self.count_of_iterations_erosion = 1
        self.kernel_type = IntVar()
        self.kernel_type.set(1)
        self.kernel = self.generate_square_kernel

        self.create_ui()

    def generate_square_kernel(self, size):
        kernel = np.ones((size, size), np.uint8)
        return kernel

    def generate_diamond_kernel(self, size):
        kernel = np.zeros((size, size), dtype=np.uint8)
        for i in range(size // 2 + 1):
            kernel[i, size // 2 - i:size // 2 + i + 1] = 1
        for i in range(size // 2 + 1, size):
            kernel[i, i - size // 2:size - (i - size // 2)] = 1
        return kernel

    def load_images(self):
        image_files = [f for f in os.listdir(self.folder_path) if os.path.isfile(os.path.join(self.folder_path, f))]
        return image_files
    def load_orig_img(self):
        image_files = [f for f in os.listdir(self.orig_img_path) if os.path.isfile(os.path.join(self.orig_img_path, f))]
        return image_files

    def create_ui(self):
        self.root.title("Image Processing")

        # Создание канвасов для изображений
        self.canvas_list = []
        for i in range(3):
            for j in range(3):
                canvas = tk.Canvas(self.frame_images, width=200, height=200)
                canvas.grid(row=i, column=j)
                self.canvas_list.append(canvas)

        # Отображение меню слева
        self.frame_menu.pack(side="left", fill="y")

        # Отображение фрейма для изображений
        self.frame_images.pack(side="right")

        # Загрузка изображений
        self.update_image()

        # Кнопка для переключения на предыдущее изображение
        prev_button = tk.Button(self.frame_menu, text="Previous Image", command=self.prev_image)
        prev_button.pack(anchor='w')

        # Кнопка для переключения на следующее изображение
        next_button = tk.Button(self.frame_menu, text="Next Image", command=self.next_image)
        next_button.pack(anchor='w')

        # Выбор типа ядра
        kernel_type_label = tk.Label(self.frame_menu, text="Kernel Type")
        kernel_type_label.pack(anchor='w')
        kernel_types = [("Square", 1), ('Diamond', 2)]
        for text, value in kernel_types:
            rb = Radiobutton(self.frame_menu, text=text, variable=self.kernel_type, value=value, command=self.update_kernel)
            rb.pack(anchor='w')

        # Ползунок для настройки размера ядра эрозии
        erosion_scale_label = tk.Label(self.frame_menu, text="Erosion Kernel Size")
        erosion_scale_label.pack(anchor='w')
        self.erosion_scale = Scale(self.frame_menu, from_=1, to=15, orient="horizontal", command=self.update_erosion)
        self.erosion_scale.set(1)
        self.erosion_scale.pack(anchor='w')

        # Ползунок для настройки размера ядра дилатации
        dilation_scale_label = tk.Label(self.frame_menu, text="Dilation Kernel Size")
        dilation_scale_label.pack(anchor='w')
        self.dilation_scale = Scale(self.frame_menu, from_=1, to=15, orient="horizontal", command=self.update_dilation)
        self.dilation_scale.set(1)
        self.dilation_scale.pack(anchor='w')

        # Ползунок для настройки количества итераций дилатации
        iterations_scale_label = tk.Label(self.frame_menu, text="Count of Iterations Dilation")
        iterations_scale_label.pack(anchor='w')
        self.iterations_scale_dilation = Scale(self.frame_menu, from_=1, to=10, orient='horizontal', command=self.update_iterations_dilation)
        self.iterations_scale_dilation.set(1)
        self.iterations_scale_dilation.pack(anchor='w')

        # Ползунок для настройки количества итераций эрозии
        iterations_scale_label = tk.Label(self.frame_menu, text="Count of Iterations Erosion")
        iterations_scale_label.pack(anchor='w')
        self.iterations_scale_erosion = Scale(self.frame_menu, from_=1, to=10, orient='horizontal', command=self.update_iterations_erosion)
        self.iterations_scale_erosion.set(1)
        self.iterations_scale_erosion.pack(anchor='w')

        # Флажок для вырезания наибольшего куска
        
        cut_the_biggest_elem=tk.Label(self.frame_menu, text='Cut the biggest element')
        cut_the_biggest_elem.pack(anchor='w')
        self.cut_the_biggest_obj=tk.Checkbutton(self.frame_menu, text='On', variable=self.enabled, command=self.update_cut)
        self.cut_the_biggest_obj.pack(anchor='w')

        save_button = tk.Button(self.frame_menu, text="Save Image", command=self.save_image)
        save_button.pack(anchor='w')

    


    def prev_image(self):
        self.current_image_index -= 1
        if self.current_image_index < 0:
            self.current_image_index = len(self.image_files) - 1
        self.update_image()

    def next_image(self):
        self.current_image_index += 1
        if self.current_image_index >= len(self.image_files):
            self.current_image_index = 0
        self.update_image()

    def update_kernel(self):
        if self.kernel_type.get() == 1:
            self.kernel = self.generate_square_kernel
        else:
            self.kernel = self.generate_diamond_kernel
        self.update_image()

    def update_dilation(self, value):
        self.kernel_size_dilation = int(value)
        self.update_image()

    def update_erosion(self, value):
        self.kernel_size_erosion = int(value)
        self.update_image()

    def update_iterations_dilation(self, value):
        self.count_of_iterations_dilation = int(value)
        self.update_image()

    def update_iterations_erosion(self, value):
        self.count_of_iterations_erosion = int(value)
        self.update_image()
    def update_cut(self):
        self.update_image()

    def apply_dilation(self, image):
        kernel = self.kernel(self.kernel_size_dilation)
        processed_image = cv2.dilate(image, kernel, iterations=self.count_of_iterations_dilation)
        return processed_image

    def apply_erosion(self, image):
        kernel = self.kernel(self.kernel_size_erosion)
        processed_image = cv2.erode(image, kernel, iterations=self.count_of_iterations_erosion)
        return processed_image

    def find_contours(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        return contours

    def get_largest_contour(self, image):
        contours = self.find_contours(image)
        if contours:
            return max(contours, key=cv2.contourArea)
        else:
            return None

    def darken_small_objects(self, image):
        largest_contour = self.get_largest_contour(image)
        
        if largest_contour is not None:
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.drawContours(mask, [largest_contour], -1, (255), thickness=cv2.FILLED)  # Используем thickness=cv2.FILLED, чтобы закрасить контур
            masked_image = cv2.bitwise_and(image, image, mask=mask)
            return masked_image
        else:
            return image

    def update_image(self):
        for i in range(9):
            image_index = (self.current_image_index + i) % len(self.image_files)
            orig_img_index=(self.current_image_index+i)%len(self.image_files)
            image_path = os.path.join(self.folder_path, self.image_files[image_index])
            orig_path=os.path.join(self.orig_img_path, self.orig_img_files[orig_img_index])
            image = cv2.imread(image_path)
            orig_image=cv2.imread(orig_path)
            if self.enabled.get():  # Проверяем состояние флажка
                processed_image = self.darken_small_objects(image)
            else:
                processed_image=image
            processed_image = self.apply_dilation(processed_image)
            processed_image = self.apply_erosion(processed_image)
            
            self.processed_images[i] = processed_image
            self.processed_orig_images[i]=orig_image

        self.display_images()

    def display_images(self):
        for i in range(9):
            orig_img=cv2.cvtColor(self.processed_orig_images[i], cv2.COLOR_BGR2RGB)
            img_rgb = cv2.cvtColor(self.processed_images[i], cv2.COLOR_BGR2RGB)
            orig_img[img_rgb==0]=1
            orig_img_pil=Image.fromarray(orig_img)
            orig_img_pil=orig_img_pil.resize((200,200))
            orig_img_tk=ImageTk.PhotoImage(image=orig_img_pil)
            self.images_for_saving[i]=orig_img
            
            #img_pil = Image.fromarray(img_rgb)
            #img_pil = img_pil.resize((200, 200))
            #img_tk = ImageTk.PhotoImage(image=img_pil)
            #img=orig_img_tk[img_tk==0]=1
            self.canvas_list[i].create_image(0, 0, anchor=tk.NW, image=orig_img_tk)
            self.canvas_list[i].img_tk = orig_img_tk
    def save_image(self):
        #Путь куда сохранять
        for i in range(len(self.image_files)):
            image_path = os.path.join(self.folder_path, self.image_files[i])
            orig_path=os.path.join(self.orig_img_path, self.orig_img_files[i])
            image = cv2.imread(image_path)
            orig_image=cv2.imread(orig_path)
            if self.enabled.get():  # Проверяем состояние флажка
                processed_image = self.darken_small_objects(image)
            else:
                processed_image=image
            processed_image = self.apply_dilation(processed_image)
            processed_image = self.apply_erosion(processed_image)
            processed_orig_image=orig_image
            processed_orig_image[processed_image==0]=1
            image_path = os.path.join(self.saving_path+'/', f'img_{i}.png')
            a=cv2.imwrite(image_path, processed_orig_image)


def main():
    image_path = './images'  # Путь до папки с оригинальными изображениями
    masks_path='./images/masks' #Путь до папки с масками
    saving_path='./images/post'# Путь куда сохранять
    root = tk.Tk()
    image_processor = ImageProcessor(root, image_path, masks_path, saving_path)
    root.mainloop()

if __name__ == "__main__":
    main()