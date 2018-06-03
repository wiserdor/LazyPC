import cv2
import keras
from keras import Model
from keras.models import Sequential
import urllib.request
import os
import numpy as np
import time


class LazyNN:

    def __init__(self):
        self.xs = None
        self.ys = None
        self.num_Of_Classes = 6
        self.dense_size = 8
        self.learning_rate = 0.0001
        self.img_num = 0
        self.mobile = self.load_mobilenet()
        self.model = None
        # self.oldx=None
        self.path = 'C:/Users/carmel/Desktop/modelpics'

    def capture(self, label):
        cam = cv2.VideoCapture(0)
        cv2.namedWindow("train")
        # img_counter = 0
        print("start in..3")
        time.sleep(1)
        print("2")
        time.sleep(1)
        print("1")
        time.sleep(1)
        print("go")

        for i in range(100):
            ret, frame = cam.read()
            cv2.imshow("train", frame)
            if not ret:
                break
            k = cv2.waitKey(1)
            if k % 256 == 27:
                print("Escape hit, closing...")
                break
            img_name = "opencv_frame_{}.png".format(self.img_num)
            cv2.imwrite(self.path + '/' + label + '/' + img_name, frame)
            print("{} written!".format(img_name))
            self.img_num += 1

        cam.release()
        cv2.destroyAllWindows()

    def addexample(self):

        label = input("which example do you want to add?\nleft=0\nright=1\nup=2\ndown=3\nstop=4\ndoing nothing=5\n ")
        self.capture(label)
        print(self.img_num)

    def load_mobilenet(self):

        mobilenet = keras.applications.mobilenet.MobileNet(input_shape=None, alpha=1.0, depth_multiplier=1,
                                                           dropout=1e-3, include_top=True, weights='imagenet',
                                                           input_tensor=None, pooling=None, classes=1000)
        return Model(inputs=mobilenet.inputs, outputs=mobilenet.get_layer('conv_pw_13_relu').output)

    def train(self):

        listing = os.listdir(self.path)
        for name in listing:
            if name.startswith('.'):
                continue
            one_pos = int(name)
            y = np.zeros(self.num_Of_Classes)
            y[one_pos] = 1
            listing2 = os.listdir(self.path + '/' + name)
            for name2 in listing2:
                if name2.startswith('.'):
                    continue
                img = cv2.imread(self.path + '/' + name + '/' + name2)
                img = cv2.resize(img, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
                np_image_data = np.asarray(img)
                np_image_data = cv2.normalize(np_image_data.astype('float'), None, -0.5, .5, cv2.NORM_MINMAX)
                np_final = np.expand_dims(np_image_data, axis=0)
                m = np.asarray(self.mobile.predict(np_final), dtype='float')
                if self.xs is None:
                    self.xs = m
                    self.ys = y

                else:
                    self.xs = np.concatenate((self.xs, m), axis=0)
                    self.ys = np.vstack((self.ys, y))

        model = Sequential([
            keras.layers.Flatten(input_shape=[7, 7, 1024]),
            keras.layers.Dense(units=self.dense_size, activation='relu',
                               kernel_initializer=keras.initializers.VarianceScaling(), use_bias=True),
            keras.layers.Dense(units=self.num_Of_Classes, activation='softmax',
                               kernel_initializer=keras.initializers.VarianceScaling(), use_bias=False),
        ])
        adam = keras.optimizers.Adam(lr=self.learning_rate)
        model.compile(loss=keras.losses.categorical_crossentropy, optimizer=adam)
        batch = self.img_num
        model.fit(x=self.xs, y=self.ys, batch_size=batch, epochs=600)
        self.model = model

    def predictme(self):

        cam = cv2.VideoCapture(0)
        cv2.namedWindow("test")
        img_counter = 0

        ret, frame = cam.read()
        cv2.imshow("test", frame)
        k = cv2.waitKey(1)

        img_name = "opencv_frame_{}.png".format(img_counter)
        img = cv2.resize(frame, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
        np_image_data = np.asarray(img)
        np_image_data = cv2.normalize(np_image_data.astype('float'), None, -0.5, .5, cv2.NORM_MINMAX)
        np_final = np.expand_dims(np_image_data, axis=0)
        m = np.asarray(self.mobile.predict(np_final), dtype='float')
        print("{} written!".format(img_name))
        output = self.model.predict_classes(m)
        print(output)
        if output == 0:
            print("left")
        elif output == 1:
            print("right")
        elif output == 2:
            print("up")
        elif output == 3:
            print("down")
        elif output == 4:
            print("stop")
        elif output == 5:
            print("doing nothing")

        cam.release()
        cv2.destroyAllWindows()
        return output

def main():
    w = webcam()
    usr = input("do you want to start adding examples? y/n")
    while (usr == 'y'):
        w.addexample()
        usr = input("do you want to keep adding?")
    usr = input("train me?\n y/n")
    if (usr == 'y'):
        w.train()
    usr = input("predict? y/n")
    if (usr == 'y'):
        w.predictme()


if __name__ == "__main__":
    main()

