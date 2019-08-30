import numpy as np
import cv2
#from img_processor import detect_and_classify
import pandas as pd

# 9200
def filter_fer_scv(csv_filename, csv_new_f, 
                    label_map = {0:'Angry', 1:'Disgust', 2:'Fear', 3:'Happy', 
                                4:'Sad', 5:'Surprise', 6:'Neutral'}):
    df = pd.read_csv(csv_filename)
    #df = df.head(500)
    df = df[8000:9200]
    use_rows = []
    for index, row in df.iterrows():
        im = np.fromstring(row['pixels'], sep=' ').astype('uint8')
        im = np.reshape(im, (48, 48))
        im = cv2.resize(im, (48*10, 48*10), cv2.INTER_AREA)
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
        im = cv2.putText(im, str(index), (5, 15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        im = cv2.putText(im, row['emotion'], (5, 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.imshow('images', im)
        k = cv2.waitKey(0)
        if k == ord('1'):   
            use_rows.append(row)

    df = pd.DataFrame(use_rows)
    df.to_csv(csv_new_f)


def filter_fer(csv_filename, csv_f_new, csv_old, thresh=8, 
                use_labels = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear'],
                labels_map = {0 : 'neutral', 1:'happiness', 2:'surprise', 3:'sadnedss', 4:'anger', 5:'disgust', 6:'fear'}):
    df = pd.read_csv(csv_filename)
    df_old = pd.read_csv(csv_old)
    img = []
    emotion = []
    use = []
    for index, row in df.iterrows():
        lbl_vals = np.array([int(row[lbl]) for lbl in use_labels])
        lbl_val_max = np.max(lbl_vals)
        if lbl_val_max >= thresh:
            emotion.append(labels_map[np.argmax(lbl_vals)])
            use.append(row['Usage'])
            image = df_old.loc[[index]]
            img.append(image['pixels'].iloc[0])
    df_new = pd.DataFrame(data={'usage':use, 'pixels':img, 'emotion':emotion})
    df_new.to_csv(csv_f_new)


if __name__ == "__main__":
    #filter_fer('data\\fer2013new.csv', 'data\\fer2013_processed.csv', 'data\\fer2013.csv')
    filter_fer_scv('data\\fer2013_processed.csv', 'data\\fer2013_filtered_9000.csv')