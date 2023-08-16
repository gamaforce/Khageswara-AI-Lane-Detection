import tensorflow as tf
import cv2, glob, os, math, time, skvideo.io
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
from tensorflow import keras
import segmentation_models as sm

# Used input
input_num = 1

# Training parameter
test_size = 0.2
random_seed = 42

# Hyperparameter
epoch = 50
batch_size = 32
learning_rate = 0.001
n_encoder_decoder = 1
initial_filter = 8
image_size = (720, 720)

# Metric Function
class MaxMeanIoU(tf.keras.metrics.MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(tf.argmax(y_true, axis=-1), tf.argmax(y_pred, axis=-1), sample_weight)

def nothing(x):
    # any operation
    pass

# Loss Function
def dice_loss(y_true, y_pred, num_classes=2):
    smooth = tf.keras.backend.epsilon()
    dice = 0
    
    for index in range(num_classes):
        y_true_f = tf.keras.backend.flatten(y_true[:,:,:,index])
        y_pred_f = tf.keras.backend.flatten(y_pred[:,:,:,index])

        intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
        union = tf.keras.backend.sum(y_true_f) + \
            tf.keras.backend.sum(y_pred_f)

        dice += (2. * intersection + smooth) / (union + smooth)

    return 1 - dice/num_classes

# Upsampling layer
def upsampling2d_nearest(x, upsampling_factor_height, upsampling_factor_width):
    w = x.shape[2] * upsampling_factor_width
    h = x.shape[1] * upsampling_factor_height

    return tf.compat.v1.image.resize_nearest_neighbor(x, (h, w))

#======================================= Create model =======================================#
def create_model():
    # Variable
    encoder_layers = []

    # Input
    input_shape = (image_size[0], image_size[1], 3)
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = inputs

    # Encoder
    for i in range(n_encoder_decoder):
        filter_number = int(2**(math.log2(initial_filter)+i))
        
        x = tf.keras.layers.Conv2D(filter_number, 3, \
            activation='relu', padding='same')(x)
        
        x = tf.keras.layers.Conv2D(filter_number, 3, \
            activation='relu', padding='same')(x)

        encoder_layers.append(x)
        x = tf.keras.layers.MaxPool2D()(x)

        print(filter_number)

    # Bridge
    filter_number = int(2**(math.log2(initial_filter)+\
        n_encoder_decoder))

    x = tf.keras.layers.Conv2D(filter_number, 3, \
        activation='relu', padding='same')(x)

    x = tf.keras.layers.Conv2D(filter_number, 3, \
        activation='relu', padding='same')(x)

    print(filter_number)

    # Decoder
    for i in reversed(range(n_encoder_decoder)):
        filter_number = int(2**(math.log2(initial_filter)+i))
        x = tf.keras.layers.Lambda(upsampling2d_nearest, \
                                   arguments={'upsampling_factor_height': 2, \
                                              'upsampling_factor_width': 2})(x)
        
        x = tf.keras.layers.Concatenate(axis=3)([x, encoder_layers[i]])

        x = tf.keras.layers.Conv2D(filter_number, 3, activation='relu', padding='same')(x)

        x = tf.keras.layers.Conv2D(filter_number, 3, activation='relu', padding='same')(x)

        print(filter_number)
    
    # Output
    outputs = tf.keras.layers.Conv2D(2, 1)(x)
    outputs = tf.keras.layers.Lambda(lambda x: tf.nn.softmax(x))(outputs)

    # Create Optimizer
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # Create Loss Function
    loss = dice_loss

    # Create Model
    model = tf.keras.models.Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer = opt, loss = loss, metrics=["accuracy", MaxMeanIoU(num_classes=2)])
    
    return model
#============================================================================================#

def get_angle(lines):
    x1,y1,x2,y2 = lines
    x,y=x1-x2, y1-y2
    angle = math.atan2(y,x)
    
    return angle

def long_line(lines):
    x1,y1,x2,y2 = lines

    x,y=x1-x2, y1-y2
    m = y/x

    x1_new = 0 
    x2_new = image_size[1]

    # Jika y1_new negatif, artinya point berada di luar image array
    y1_new = m * (x1_new - x1) + y1
    if y1_new < 0 or y1_new > image_size[0]:
        y1_new = 0
        x1_new = (y1_new - y1) / m + x1

    y2_new = m * (x2_new - x1) + y1
    if y2_new < 0 or y2_new > image_size[0]:
        y2_new = 0
        x2_new = (y2_new - y1) / m + x1

    x1_new = int(x1_new)
    x2_new = int(x2_new)
    y1_new = int(y1_new)
    y2_new = int(y2_new)

    new_point = [x1_new, y1_new, x2_new, y2_new]

    return new_point

# Hough Transform Function
def hough_transform(edge, out):
    # Variables
    x1_arr = []
    x2_arr = []
    y1_arr = []
    y2_arr = []
    
    edge = cv2.cvtColor(edge, cv2.COLOR_RGB2GRAY)
    lines = cv2.HoughLinesP(edge, rho=1, theta=np.pi/180, threshold=100, minLineLength=100, maxLineGap=50)

    for line in lines:
        sudut = get_angle(line[0])
        sudut = abs(sudut)

        x1,y1,x2,y2 = line[0]
        if sudut < 2.9 :
            # Filter out the lines in the top op the image
            if (y1>50 or y2>50): 
                if (x1>10 and x1 <710) or (x2>10 and x2<710):
                    
                    x1_arr.append(x1)
                    x2_arr.append(x2)
                    y1_arr.append(y1)
                    y2_arr.append(y2)
                    cv2.line(out, (x1,y1), (x2,y2), (255,0,0), 3)
    
    return out

# Contour Function
def detect_contour(img, img_ori):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    contours, hierarchy = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    img_zero = np.zeros((img.shape)) 
    lar_idx = 0
    lar_area = 0
    sec_lar_idx = 0
    sec_lar_area = 0

    area_tol = cv2.getTrackbarPos("Area Tolerance", "Trackbars")

    for index in range(len(contours)):
        area = cv2.contourArea(contours[index])

        if area > lar_area:
            if area > area_tol:
                lar_idx = index
                lar_area = area

    for index in range(len(contours)):
        area = cv2.contourArea(contours[index])

        if lar_area > area and sec_lar_area < area:
            if area > area_tol:
                sec_lar_idx = index
                sec_lar_area = area

        # if  area > 55000:
        #     cv2.drawContours(img_zero, contours[index], -1, (255,255,255), 3)
    if lar_idx != 0:
        cv2.drawContours(img_zero, contours[lar_idx], -1, (255,255,255), 3)
        cv2.drawContours(img_zero, contours[sec_lar_idx], -1, (255,255,255), 3)

        cv2.drawContours(img_ori, contours[lar_idx], -1, (36,255,12), 3)
        cv2.drawContours(img_ori, contours[sec_lar_idx], -1, (36,255,12), 3)

        # epsilon = 1 * cv2.arcLength(contours[lar_idx], True)
        # approx = cv2.approxPolyDP(contours[lar_idx], epsilon, True)
        # cv2.drawContours(img_zero, [approx], -1, (255,255,255), 3)
        # cv2.drawContours(img_ori, [approx], -1, (36,255,12), 3)

        # epsilon = 1 * cv2.arcLength(contours[sec_lar_idx], True)
        # approx = cv2.approxPolyDP(contours[sec_lar_idx], epsilon, True)
        # cv2.drawContours(img_zero, [approx], -1, (255,255,255), 3)
        # cv2.drawContours(img_ori, [approx], -1, (36,255,12), 3)
        

    # print(sec_lar_idx)
    # print(lar_idx)
    return img_zero, img_ori

# Fill Function
def fill_area(img):
    # Set Image to Gray
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    # Fill Area
    cv2.fillPoly(img, cnts, [255,255,255])

    return img

# Variables
masks_aspalt = []
masks_edge =[]

model = create_model()

model_path = os.path.join("model/model_1.h5")
model.load_weights(model_path)

tf.debugging.set_log_device_placement(True)

# Video Writer
outputfile = "video/output/output_video_"+ str(input_num) +".mp4"
size = (720, 720)
fps = 20
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
writer = cv2.VideoWriter(outputfile, fourcc, fps, size)

# Load Video
video_path = os.path.join("video/input/input_video_"+ str(input_num) +".mp4")
cap = cv2.VideoCapture(video_path)

# Create Trackbars
cv2.namedWindow("Trackbars")
cv2.createTrackbar("Treshold", "Trackbars", 10, 100, nothing)
cv2.createTrackbar("Lower Limit", "Trackbars", 25, 179, nothing)
cv2.createTrackbar("Upper Limit", "Trackbars", 87, 179, nothing)
cv2.createTrackbar("Area Tolerance", "Trackbars", 50000, 90000, nothing)

while(cap.isOpened()):
    start_time = time.time()
    ret, frame = cap.read()
    # frame = cv2.imread("training_dataset/images/1_11.jpg")
    if True:
        image_height = frame.shape[0]
        image_width = frame.shape[1]
        frame = frame[0:image_height, (image_width-image_height)//2:(image_width-image_height)//2+image_height]
        frame_ori = frame.copy()
        frame = cv2.resize(frame, image_size)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.normalize(frame, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)   

        # Predict mask
        pred = model.predict(np.expand_dims(frame, 0))

        range_tresh = cv2.getTrackbarPos("Treshold", "Trackbars") / 10000

        # Process mask
        mask = pred.squeeze()
        mask = np.stack((mask,)*3, axis=-1)
        mask[mask >= range_tresh] = 255
        mask[mask < range_tresh ] = 0

        mask_aspalt = mask[:, :, 1]
        mask_aspalt = np.uint8(mask_aspalt)

        lower_hue = cv2.getTrackbarPos("Lower Limit", "Trackbars")
        upper_hue = cv2.getTrackbarPos("Upper Limit", "Trackbars")

        lower_green = np.array([lower_hue,0,0], dtype=np.uint8)
        upper_green = np.array([upper_hue,255,255], dtype=np.uint8)

        # Threshold the HSV image to get only green colors
        hsv = cv2.cvtColor(frame_ori, cv2.COLOR_BGR2HSV)
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        mask_green = cv2.bitwise_not(mask_green)
        mask_green = cv2.resize(mask_green, image_size)
        mask_green = np.stack((mask_green,)*3, axis=-1)

        mask_all = cv2.bitwise_and(mask_aspalt, mask_green)

        # Process Edge
        edge, frame_ori = detect_contour(mask_all, frame_ori)
        edge = np.uint8(edge)

        # Edge Filled
        edge_filled = fill_area(edge)

        # Hough Transform
        # frame_ori = hough_transform(edge, frame_ori)
    
        masks_aspalt.append(mask_aspalt)
        masks_edge.append(edge)

        #Show Video
        cv2.namedWindow("Model Predict", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("Model Predict", cv2.WND_PROP_FULLSCREEN, cv2.WND_PROP_FULLSCREEN)
        cv2.imshow("Model Predict", mask_aspalt)

        cv2.namedWindow("Contour", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("Contour", cv2.WND_PROP_FULLSCREEN, cv2.WND_PROP_FULLSCREEN)
        cv2.imshow("Contour", edge)

        cv2.namedWindow("Road", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("Road", cv2.WND_PROP_FULLSCREEN, cv2.WND_PROP_FULLSCREEN)
        cv2.imshow("Road", frame_ori)

        cv2.namedWindow("Filled", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("Filled", cv2.WND_PROP_FULLSCREEN, cv2.WND_PROP_FULLSCREEN)
        cv2.imshow("Filled", edge_filled)

        cv2.namedWindow("Mask Green", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("Mask Green", cv2.WND_PROP_FULLSCREEN, cv2.WND_PROP_FULLSCREEN)
        cv2.imshow("Mask Green", mask_green)

        writer.write(frame_ori)

        # Stop when "q" is pressed
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    else:
        break

# Stop Video
print("Video Ended")
writer.release()
cap.release()
cv2.destroyAllWindows()

