import urllib.request
import numpy as np
import cv2 
import pytesseract
import os
import io
import re
from flask import Flask, request, redirect, jsonify,send_file
from werkzeug.utils import secure_filename
from google.cloud import vision
from google.cloud.vision import types

ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])
UPLOAD_FOLDER = 'C:/Users/hp/Desktop/gitBitirme/Images/'
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/images', methods=['POST'])
def upload():
	# check if the post request has the file part
	if 'file' not in request.files:
		resp = jsonify({'message' : 'No file part in the request'})
		resp.status_code = 400
		return resp
	file = request.files['file']
	if file.filename == '':
		resp = jsonify({'message' : 'No file selected for uploading'})
		resp.status_code = 400
		return resp
	if (file and allowed_file(file.filename)):
		filename = secure_filename(file.filename)
		# file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
		resp = jsonify({'message' : 'File successfully uploaded'})
		resp.status_code = 201
		return resp
	else:
		resp = jsonify({'message' : 'Allowed file types are txt, pdf, png, jpg, jpeg, gif'})
		resp.status_code = 400
		return resp

@app.route('/images/file-upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    if (file and allowed_file(file.filename)):
        filename = secure_filename(file.filename)
        name1 = request.files['file']
        # path=os.path.dirname(os.path.abspath(name1))
        # file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        npimg = np.fromfile(file,np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
      
        result = imagePreProcessing(img)
        #o.ocr_operation()
        resp = jsonify({'message' : 'File successfully uploaded'},result)
        resp.status_code = 201
        return resp

# def processing(img):
#     res = imagePreProcessing(img)
#     return res  


def imagePreProcessing(img):
    #convert string data to numpy array
    # npimg = np.fromstring("./Images/1.jpg", np.uint8)
    # # convert numpy array to image
    # img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    # img = cv2.imread("./Images/1.jpg")
    # print(img)

    # if(img.any()):
    #     print("fsdfsd")
    source = img
   
    img_width = img.shape[1]
    img_height = img.shape[0]
    for i in range(3):
        dst = cv2.GaussianBlur(source,(0,0),50)
        dst2 = cv2.addWeighted(source,1.50,dst,-0.50,0)
        source = dst2
    
    ret,result = cv2.threshold(source, 1, 255, cv2.THRESH_BINARY)
    # cv2.imwrite("./Images/imagePreProcessingResult.jpg",result)

    res = detectBoxesAndFillCoordinateLists(result,source)
    return res
    
    

def detectBoxesAndFillCoordinateLists(img,source):

    img_width = img.shape[1]
    img_height = img.shape[0]    
    startYList = []
    startXList = []
    heightList = []
    widthList = []
    img_blob = cv2.dnn.blobFromImage(img, 1/255, (416,608), swapRB=True, crop=False)
    labels = ["text"]
    
    colors = ["125,150,130", "125,255,130", "125,255,130"]
    colors = [np.array(color.split(",")).astype("int") for color in colors]
    colors = np.array(colors)
    colors = np.tile(colors,(18,1))
    
    model = cv2.dnn.readNetFromDarknet("C:/Users/hp/Desktop/Bitirme/yolov4.cfg", "C:/Users/hp/Desktop/Bitirme/yolov4.weights")
    layers = model.getLayerNames()
    output_layer = [ layers[layer[0]-1] for layer in model.getUnconnectedOutLayers()]
    
    model.setInput(img_blob)
    
    detection_layers = model.forward(output_layer)
    for detection_layer in detection_layers: 
        for object_detection in detection_layer:
            scores = object_detection[5:]
            predicted_id = np.argmax(scores)
            confidence = scores[predicted_id]
            if confidence > 0.10: 
                label = labels[predicted_id]
                bounding_box = object_detection[0:4] * np.array([img_width,img_height,img_width,img_height])
                (box_center_x, box_center_y, box_width, box_height) = bounding_box.astype("int")
                
                start_x = int(box_center_x - (box_width/2))
                start_y = int(box_center_y - (box_height/2))
                
                end_x = start_x + box_width
                end_y = start_y + box_height
                
                heightList.append(box_height)
                widthList.append(box_width)
                startYList.append(start_y)
                startXList.append(start_x)
    
                
                box_color = colors[0]
                box_color = [int(each) for each in box_color]
               
    # cv2.rectangle(img, (start_x,start_y),(end_x,end_y),box_color,1)
    # cv2.putText(img,label,(start_x,start_y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)
    # cv2.imwrite("./Images/file.jpg",img)
    
    result = croppingProcess(heightList,widthList,startYList,startXList,img,img_width,img_height,source)
    return result            
    

def croppingProcess(heightList,widthList,startYList,startXList,img,img_width,img_height,source):
    for i in range(len(startYList)):
        minIndex = i
        for j in range(i+1,len(startYList)):
            if startYList[minIndex] > startYList[j]:
                    minIndex = j
    
        startYList[i], startYList[minIndex] = startYList[minIndex], startYList[i]
        startXList[i], startXList[minIndex] = startXList[minIndex], startXList[i]
        widthList[i], widthList[minIndex] = widthList[minIndex], widthList[i]
        heightList[i], heightList[minIndex] = heightList[minIndex], heightList[i]
    
    croppedImageIndex = 0
    tempValue = 0
    for i in range(len(startYList)-1):
        if startYList[i+1] - (startYList[i]+heightList[i]) > 25 and croppedImageIndex < 5:
            crop_img = img[tempValue: startYList[i]+heightList[i]+10, 0:img_width]
            path = "C:/Users/hp/Desktop/Images/FirstImage/yeniKirpilmiskirpilmisFoto{}.jpg".format(croppedImageIndex)
            cv2.imwrite(path, crop_img)
            croppedImageIndex += 1
            tempValue = startYList[i]+heightList[i] 
    image=source
    result = ocr_operation(image)
    return result
    #cv2.imshow("Detection window", img)
    #cv2.imwrite("C:/Users/yasar/Desktop/fisler/sonEgitim/deneme/denemeeeeeee.jpg",img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

# img = imagePreProcessing()
# detectBoxesAndFillCoordinateLists()
# croppingProcess()
                
receipt = {"MarketName":"","Date":"","items":[],"Total":""}

def ocr_operation(img):
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'C:/Users/hp/Desktop/Bitirme/API/fisproject-312420-23eb712079e0.json'
    my_path = 'C:/Users/hp/Desktop/Bitirme/Images/FirstImage'
    any_digit = '()'
    os.chdir(my_path) #Just to be sure you are in your working directory not else where!
    array = []
    client = vision.ImageAnnotatorClient()
    success, encoded_image = cv2.imencode('.jpg', img)
    roi_image = encoded_image.tobytes()
    roi_image = vision.types.Image(content=roi_image)
    response = client.annotate_image({'image':roi_image, 'features': [{'type': vision.enums.Feature.Type.FACE_DETECTION,'max_results':40}],})
    response_text = client.text_detection(image=roi_image)
    for r in response_text.text_annotations:
        d = {
            'text': r.description
        }
        array.append(r.description)
        print(r.description)
        break

    array = []
    text = ""

    for f in os.listdir('.'):
        if f.endswith('.jpg'):
            image = cv2.imread(f)
            configuration = ("--psm 6")
            newText = pytesseract.image_to_string(image, config=configuration,lang = 'eng+tur')
            text = text + " "+ newText
    print(text)

    marketName = findMarketName(text)
    receipt["MarketName"] = marketName


    date = find_date(text)
    receipt["Date"] = date

    amounts = find_amounts(text)
    receipt["Total"]: amounts

    return receipt
        
def find_date(text):
    date_pattern = r'(0[1-9]|[12][0-9]|3[01])[\.|-|/](0[1-9]|1[012])[\.|-|/](19|20)\d\d'
    try:
        date = re.search(date_pattern, text).group()
        return date
    except:
        return None    

def find_amounts(text):
    lines_with_chf = []
    splits = text.splitlines()
    for line in splits:
        if (re.search(r'\*',line) ) or find_number(line) or (re.search(r'\%',line)):
         lines_with_chf.append(line)
    print(lines_with_chf)

    items = []
    for line in lines_with_chf:
        print(line)
        if re.search(r'KDV',line):
            continue
        if re.search(r'TOPLAM', line) or re.search(r'TOP', line):
            receipt["Total"] = line
        else:
            if( re.search(r'NAKİT', line) or re.search(r'PARA', line)or re.search(r'TARİH', line) or re.search(r'TARIH', line)):
                continue
            else:
                items.append(line)
        
        receipt["items"] = items


def findMarketName(text):
    splits = text.splitlines()
    market_name = splits[0] + ' ' + splits[1]
    return market_name

def find_number(line):
    amounts = re.findall(r"\d{1,2}[\,\.]{1}\d{1,2}", line.strip())
    if not amounts :
        return False
    else: 
        return True

if __name__ == '__main__':
    app.run()
