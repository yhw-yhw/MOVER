import cv2

def debug_project_bbox(self):
    debug_img = np.copy(self.image)
    cnt=0
    for one in det_bbox.detach().cpu().numpy():
        cv2.putText(debug_img, f'{cnt}', (int(one[0]), int(one[1])), cv2.FONT_HERSHEY_COMPLEX,0.5,(255,0,0),1)
        cv2.rectangle(debug_img, (int(one[0]), int(one[1])), 
                    (int(one[2]), int(one[3])), (255, 0, 0))
        cnt+=1
    cnt=0
    for one in torch.cat(proj_bbox).detach().cpu().numpy():
        cv2.putText(debug_img, f'{cnt}', (int(one[0]), int(one[1])), cv2.FONT_HERSHEY_COMPLEX,0.5,(0,255,0),1)
        cv2.rectangle(debug_img, (int(one[0]), int(one[1])), 
                    (int(one[2]), int(one[3])), (0, 255, 0))
        cnt+=1
    cv2.imshow('debug', debug_img)
    cv2.waitKey(1)
    # cv2.destroyAllWindows()