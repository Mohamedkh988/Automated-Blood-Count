import cv2
import time
from utils import iou
from scipy import spatial
from darkflow.net.build import TFNet
import PySimpleGUIQt as sG
import os.path

options = {'model': 'cfg/tiny-yolo-voc-3c.cfg',
           'load': 3750,
           'threshold': 0.1,
           'gpu': 0.7}

tfnet = TFNet(options)

pred_bb = []  # predicted bounding box
pred_cls = []  # predicted class
pred_conf = []  # predicted class confidence


def blood_cell_count(file_name):
    global rbc
    global wbc
    global platelets
    rbc = 0
    wbc = 0
    platelets = 0

    cell = []
    cls = []
    conf = []

    record = []
    tl_ = []
    br_ = []
    iou_ = []
    iou_value = 0

    tic = time.time()
    image = cv2.imread('data/' + file_name)
    output = tfnet.return_predict(image)

    for prediction in output:
        label = prediction['label']
        confidence = prediction['confidence']
        tl = (prediction['topleft']['x'], prediction['topleft']['y'])
        br = (prediction['bottomright']['x'], prediction['bottomright']['y'])

        if label == 'RBC' and confidence < .5:
            continue
        if label == 'WBC' and confidence < .25:
            continue
        if label == 'Platelets' and confidence < .25:
            continue
        # clearing up overlapped same platelets
        if label == 'Platelets':
            if record:
                tree = spatial.cKDTree(record)
                index = tree.query(tl)[1]
                iou_value = iou(tl + br, tl_[index] + br_[index])
                iou_.append(iou_value)

            if iou_value > 0.1:
                continue

            record.append(tl)
            tl_.append(tl)
            br_.append(br)

        center_x = int((tl[0] + br[0]) / 2)
        center_y = int((tl[1] + br[1]) / 2)
        center = (center_x, center_y)

        if label == 'RBC':
            color = (255, 0, 0)
            rbc = rbc + 1
        if label == 'WBC':
            color = (0, 255, 0)
            wbc = wbc + 1
        if label == 'Platelets':
            color = (0, 0, 255)
            platelets = platelets + 1

        radius = int((br[0] - tl[0]) / 2)
        image = cv2.circle(image, center, radius, color, 2)
        font = cv2.FONT_HERSHEY_COMPLEX
        image = cv2.putText(image, label, (center_x - 15, center_y + 5), font, .5, color, 1)
        cell.append([tl[0], tl[1], br[0], br[1]])

        if label == 'RBC':
            cls.append(0)
        if label == 'WBC':
            cls.append(1)
        if label == 'Platelets':
            cls.append(2)

        conf.append(confidence)

    toc = time.time()
    pred_bb.append(cell)
    pred_cls.append(cls)
    pred_conf.append(conf)
    avg_time = (toc - tic) * 1000
    print('{0:.5}'.format(avg_time), 'ms')

    cv2.imwrite('output/' + file_name, image)
    cv2.imshow('Total RBC: ' + str(rbc) + ', WBC: ' + str(wbc) + ', Platelets: ' + str(platelets), image)
    print('Press "ESC" to close . . .')
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()


# window layout in 2 columns
file_list_column = [
    [
        sG.Text("Name"),
        sG.In(size=(25, 1), enable_events=True, key="-NAME-"),
    ],
    [
        sG.Text("Age"),
        sG.In(size=(25, 1), enable_events=True, key="-AGE-"),
    ],
    [
        sG.Text("Gender"),
        sG.In(size=(25, 1), enable_events=True, key="-GENDER-"),
    ],
    # [
    #     sG.Text("PCV"),
    #     sG.In(size=(25, 1), enable_events=True, key="-PCV-"),
    # ],
    [
        sG.Text("Image Folder"),
        sG.In(size=(18.5, 1), enable_events=True, key="-FOLDER-"),
        sG.FolderBrowse(size=(6, 1)),
    ],
    [
        sG.Listbox(
            values=[], enable_events=True, size=(35, 10), key="-FILE LIST-"
        )
    ],
]
# For now will only show the name of the file that was chosen
results_viewer_column = [
    [sG.Text("Choose an image from list on left:")],
    [sG.Text(size=(40, 1), key="-TOUT-")],
    [sG.Image(key="-IMAGE-")],
]
results_viewer_column2 = [
    [sG.Text("Low RBC count may be caused by deficiency in iron, vitamin B6 or vitamin B12.\nIt can also signify "
             "internal bleeding, kidney disease, or malnutrition.\n\nHigh RBC count may be caused by smoking, "
             "congenital heart disease, dehydration,\nhypoxia or pulmonary fibrosis.\n\nLow WBC count may be caused by "
             "cancer, autoimmune disorders, Crohn's disease,\nmalnutrition, rheumatoid arthritis, radiation "
             "poisoning, liver damage, an infection or bone marrow damage.\n\nHigh WBC count may be caused by an "
             "infection, immunosuppression, bone marrow disorders, leukemia,\nopen wounds, inflammation, stress, "
             "trauma, labor, pregnancy, smoking, obesity, allergies or excessive exercise.\n\nLow Platelet count may "
             "be caused by platelets being trapped in an enlarged spleen,\ndecreased platelet production rate or "
             "increased breakdown of platelets.\n\nHigh Platelet count may be caused by anemia, cancer, infection, "
             "inflammation or a splenectomy")],
    [sG.Text(size=(40, 3), key="-TOUT2-")],
    [sG.Button(button_text="Save", key="-SAVE-", size=(6, 1))]
]
# ----- Full layout -----
layout = [
    [
        sG.Column(file_list_column),
        sG.VSeperator(),
        sG.Column(results_viewer_column),
        sG.VSeperator(),
        sG.Column(results_viewer_column2),
    ]
]
window = sG.Window("Automated Blood Smear", layout)
# Run the Event Loop
while True:
    event, values = window.read()
    name = values["-NAME-"]
    age = values["-AGE-"]
    gender = values["-GENDER-"]
    if event == "Exit" or event == sG.WIN_CLOSED:
        break
    # Folder name was filled in, make a list of files in the folder
    if event == "-FOLDER-":
        folder = values["-FOLDER-"]
        try:
            # Get list of files in folder
            file_list = os.listdir(folder)
        except:
            file_list = []
        fnames = [
            f
            for f in file_list
            if os.path.isfile(os.path.join(folder, f))
               and f.lower().endswith((".png", ".gif", ".jpeg", ".jpg"))
        ]
        window["-FILE LIST-"].update(fnames)
    elif event == "-FILE LIST-":  # A file was chosen from the listbox
        try:
            filename = os.path.join(values["-FOLDER-"], values["-FILE LIST-"][0])
            imagename = values["-FILE LIST-"][0]
            window["-TOUT-"].update(filename)
            window["-IMAGE-"].update(filename=filename)
            blood_cell_count(imagename)
            # if platelets != 0:
            #     results = "Results:\n" + "Red Blood Cell Count: " + str(
            #         rbc * 7 * 5 * 10000) + "RBCs/uL (Normal: 4.5-6.5 " \
            #                                "million)" + "\nWhite " \
            #                                             "Blood Cell " \
            #                                             "Count: " + \
            #               str(wbc * 7 * 4 * 50) + " WBCs/uL  (Normal: 4-11 thousand)" + "\nPlatelet Count: " + str(
            #         platelets * 7 * 25 * 2000) + " platelets/μL        (Normal: 150-450 thousand) "
            #     window["-TOUT2-"].update(results)
            # else:
            results = "Results:\n" + "Red Blood Cell Count: " + str(
                    rbc * 7 * 5 * 10000) + "RBCs/uL (Normal: 4.5-6.5 " \
                                           "million)" + "\nWhite " \
                                                        "Blood Cell " \
                                                        "Count: " + \
                          str(wbc * 7 * 4 * 50) + " WBCs/uL  (Normal: 4-11 thousand)" + "\nPlatelet Count: " + str(platelets * 7 * 25 * 2000) + " platelets/uL (Normal: 150-450 thousand)"
            window["-TOUT2-"].update(results)
            if platelets == 0:
                window["-TOUT2-"].update(results + "\nPlease repeat test with images that contain platelets.")
            elif rbc == 0:
                window["-TOUT2-"].update(results + "\nPlease repeat test with images that contain red blood cells.")
            elif wbc == 0:
                window["-TOUT2-"].update(results + "\nPlease repeat test with images that contain white blood cells")
        except:
            pass
    elif event == "-SAVE-":
        try:
            s = open("Results.txt", "w")
            s.write("Name: " + name + "\n" + "Age: " + age + "\n" + "Gender: " + gender + "\n" + results)
            s.close()
        except:
            pass
window.close()
