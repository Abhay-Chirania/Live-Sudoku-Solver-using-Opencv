import numpy as np
import cv2
import imutils
import tensorflow as tf


    ##TODO: TRAIN THE MODEL BETTER


model=tf.keras.models.load_model("digitRecognization.h5")
circles=[]
c=0
def mousePoints(event,x,y,flag,params):
    global c,circles
    if event==cv2.EVENT_LBUTTONDOWN:
        circles.append([x,y])
        c+=1
        if c==4:
            circles.sort(key=lambda x:x[1])
            c1=circles[:2]
            c1.sort(key=lambda x:x[0])
            c2=circles[2:]
            c2.sort(key=lambda x:x[0])
            circles=c1+c2
def getROI(img):
    width,height=270,270
    while True:
        if c==4:
            ps1=np.float32([circles[0],circles[1],circles[2],circles[3]])
            ps2=np.float32([[0,0],[width,0],[0,height],[width,height]])
            matrix=cv2.getPerspectiveTransform(ps1,ps2)
            imgCrop=cv2.warpPerspective(img,matrix,(width,height))
            return imgCrop
        for x in range(len(circles)):
            if len(circles)>0:
                cv2.circle(img,(circles[x][0],circles[x][1]),5,(0,255,0),-1)
        cv2.imshow("Image",img)
        cv2.setMouseCallback("Image",mousePoints)
        cv2.waitKey(1)
def findBiggestRect(rects):
    newRects=[]
    for i in range(0,270,30):
        rngY=(i,i+30)
        for j in range(0,270,30):
            rngX=(j,j+30)
            holder=[]
            for rect in rects:
                x,y=rect[0],rect[1]
                if rngX[0]<=x<=rngX[1] and rngY[0]<=y<=rngY[1]:
                    holder.append(rect)
            if holder:
                t=max(holder,key=lambda x: x[2]*x[3])
                newRects.append(t)
    return newRects


def solveSudoku(sudoku):
    board=sudoku
    def solve(bo):
        find = find_empty(bo)
        if not find:
            return True
        else:
            row, col = find

        for i in range(1,10):
            if valid(bo, i, (row, col)):
                bo[row][col] = i

                if solve(bo):
                    return True

                bo[row][col] = 0

        return False


    def valid(bo, num, pos):
        # Check row
        for i in range(len(bo[0])):
            if bo[pos[0]][i] == num and pos[1] != i:
                return False

        # Check column
        for i in range(len(bo)):
            if bo[i][pos[1]] == num and pos[0] != i:
                return False

        # Check box
        box_x = pos[1] // 3
        box_y = pos[0] // 3

        for i in range(box_y*3, box_y*3 + 3):
            for j in range(box_x * 3, box_x*3 + 3):
                if bo[i][j] == num and (i,j) != pos:
                    return False

        return True


    def print_board(bo):
        for i in range(len(bo)):
            if i % 3 == 0 and i != 0:
                print("- - - - - - - - - - - - - ")

            for j in range(len(bo[0])):
                if j % 3 == 0 and j != 0:
                    print(" | ", end="")

                if j == 8:
                    print(bo[i][j])
                else:
                    print(str(bo[i][j]) + " ", end="")


    def find_empty(bo):
        for i in range(len(bo)):
            for j in range(len(bo[0])):
                if bo[i][j] == 0:
                    return (i, j)  # row, col

        return None

    print_board(board)
    solve(board)
    print("___________________")
    print_board(board)
    return board
def main(imgPath):       
    sudoku=[[0 for _ in range(9)]for _ in range(9)]
    img=cv2.imread(imgPath)
    img=cv2.resize(img,(512,512),cv2.INTER_AREA)
    cv2.putText(img,"Click on the 4 corner of the Sudoku",(10,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1,cv2.LINE_AA)
    imgCrop=getROI(img)
    gray=cv2.cvtColor(imgCrop,cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (1, 1), 0)
    threshold=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,125,1)
    ctrs, hier = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    rects = [cv2.boundingRect(ctr) for ctr in ctrs if 10<cv2.contourArea(ctr)<500]
    rects=findBiggestRect(rects)
    if len(rects)==0:
        print("Sorry No Data Found. Either Image is unclear or the background is dark")
    else:
        for rect in rects:
            if rect[2]>80 or rect[3]>80:
                continue
            #cv2.rectangle(imgCrop, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3) 
            leng = int(rect[3] * 1.6)
            pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
            pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
            try:
                roi = threshold[pt1:pt1+leng, pt2:pt2+leng]
                roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
                roi = cv2.dilate(roi, (3, 3))
                roi=np.array(roi)
                roi=roi/255.0
                roi=roi.reshape(-1,28,28,1)
                x=model.predict(roi)
                p=list(x[0])
                predicted_num=p.index(max(p))
                if predicted_num!=0:
                    #cv2.putText(imgCrop,str(predicted_num),(rect[0]-10,rect[1]),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),1,cv2.LINE_AA)
                    p=0
                    for i in range(0,270,30):
                        rngY=(i,i+30)
                        q=0
                        for j in range(0,270,30):
                            rngX=(j,j+30)
                            if rngX[0]<=rect[0]<=rngX[1] and rngY[0]<=rect[1]<=rngY[1]:
                                sudoku[p][q]=predicted_num
                            q+=1
                        p+=1
                            
            except:
                continue
    cv2.imshow("th",threshold)
    solved=solveSudoku(sudoku)
    solvedImg=imgCrop
    for i in range(len(solved)):
        for j in range(len(solved[i])):
            cv2.putText(solvedImg,str(solved[i][j]),((j*30)+5,(i*30)+20),cv2.FONT_HERSHEY_SIMPLEX,0.65,(255,0,0),1,cv2.LINE_AA)
    cv2.imshow("SolvedImage",solvedImg)



imgPath="sudoku.jpg"
if __name__=='__main__':
    main(imgPath)
