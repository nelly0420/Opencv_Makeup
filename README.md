# 프로젝트 시작

OpenCV와 dlib을 사용하여 얼굴 메이크업 앱을 개발하는 프로젝트입니다.

---

# **Virtual Makeup App using OpenCV and dlib**

![Virtual Makeup Banner](https://via.placeholder.com/1200x400.png?text=Virtual+Makeup+App+Banner)

This project leverages **OpenCV** and **dlib** to create a virtual makeup application that allows users to apply lipstick, blush, eyeliner, and eyeshadow to their photos. The application uses facial landmark detection to accurately place makeup on the appropriate facial regions.

## **Features**

- **Lipstick Application:** Apply different shades of lipstick.
- **Blush Application:** Add blush to the cheeks with various colors.
- **Eyeliner:** Customize eyeliner styles and colors.
- **Eyeshadow:** Apply eyeshadow with options for multiple colors and gradients.
- **Real-time Makeup:** Apply makeup in real-time using a webcam feed.
- **Save Results:** Save the final makeup-applied image.

## **Demo**

![Demo GIF](https://via.placeholder.com/800x400.png?text=Demo+GIF)

## **Installation**

### **Prerequisites**

- Python 3.8 or higher
- pip (Python package installer)

### **Setup**

1. **Clone the repository:**

   c:\project

2. **Install the required packages:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Download the dlib pretrained model:**

   Download the [shape_predictor_68_face_landmarks.dat](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2) file and place it in the project directory.

## **Usage**

### **Running the Application**

To run the application, execute the following command:

```bash
python app.py
```

### **Applying Makeup**

- **Lipstick:** Select the color from the lipstick palette and apply.
- **Blush:** Choose the blush shade and intensity.
- **Eyeliner:** Pick a style and color for the eyeliner.
- **Eyeshadow:** Select a color or gradient for the eyeshadow.

### **Saving Your Work**

After applying the desired makeup, you can save the image by clicking the **Save** button.

## **How It Works**

### **Facial Landmark Detection**

The app uses dlib's 68-point facial landmark detector to identify key facial features:
- **Lips**
- **Eyes**
- **Cheeks**

These landmarks guide the placement of makeup on the user's face.

### **Makeup Application**

OpenCV is used to:
- **Overlay Colors:** Apply colors to the lips, cheeks, and eyes.
- **Blend Effects:** Smoothly blend the applied makeup with the original image.

## **File Structure**

```
virtual-makeup-app/
├── app.py                   # Main application file
├── requirements.txt         # Python dependencies
├── shape_predictor_68_face_landmarks.dat # dlib model file
├── static/                  # Directory for static assets (images, etc.)
├── template/                # Html files
└── README.md                # Project documentation
```

## **Dependencies**

- OpenCV
- dlib
- NumPy
- Flask (if using a web-based interface)

