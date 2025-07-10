import argparse
import gc
import cv2
import json
import logging
import numpy as np
from pathlib import Path
from sklearn.metrics import jaccard_score

import torch
from torch.utils.data import DataLoader

import albumentations as album
import segmentation_models_pytorch as smp

from glob import glob
from collections import defaultdict
import json



class FloorPlanTestDataset(torch.utils.data.Dataset):    
    def __init__(
            self, 
            img_path_list, 
            cls_cnt=13, 
            augmentation=None, 
            preprocessing=None,
    ):
        self.img_path_list = img_path_list
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        
        self.cls_cnt = cls_cnt
        tmp_array = np.zeros((self.cls_cnt+1, self.cls_cnt), int)
        np.fill_diagonal(np.fliplr(tmp_array), 1, wrap=True)
        
        self.ohe_array = tmp_array[::-1]
        self.subcat_dict = {
            1: 1, 2: 2, 3: 3, 13: 4, 14: 5, 15: 6, 16: 7, 
            17: 8, 18: 9, 19: 10, 20: 11, 22: 12, 23: 13
        }


        
    def __getitem__(self, i):
        
        img_path = str(self.img_path_list[i])
        
        
        # read images and resize 
        # 테스트 데이터 사이즈 줄이기
        simg = cv2.imread(img_path, cv2.IMREAD_COLOR)
        simg = cv2.cvtColor(simg, cv2.COLOR_BGR2RGB)

        target_h, target_w = 3508, 4963
        h, w = simg.shape[:2]
        if (h, w) != (target_h, target_w):
            gray = cv2.cvtColor(simg ,cv2.COLOR_BGR2GRAY)
            gray_3= cv2.merge([gray, gray, gray])
            # 원본 종횡비 유지
            scale = min(target_w / w, target_h / h)/2
            new_w = int(w * scale)
            new_h = int(h * scale)
            resize_img = cv2.resize(gray_3, (new_w, new_h), interpolation = cv2.INTER_AREA)
            
            # 패딩 추가
            delta_w = target_w - new_w
            delta_h = target_h - new_h
            top, bottom = delta_h // 2, delta_h - (delta_h // 2)
            left, right = delta_w // 2, delta_w - (delta_w // 2)

            # 원본 이미지의 (0, 0) 위치의 색상 추출
            border_color = simg[0, 0].tolist()

            simg = cv2.copyMakeBorder(resize_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=border_color)
            

        #cv2.imwrite(f'resoze_image{i}.png',simg)
        #print('simgshape', simg.shape)

        bh, bw = simg.shape[:2]
        if bh > bw:
            w, h = 436, 620
            fw, fh = 448, 640
        else:
            w, h = 620, 436
            fw, fh = 640, 448

        croped_img = simg[:bh // 32 * 32,:bw // 32 * 32,:]

        reduce_ratio_x = w / croped_img.shape[1]
        reduce_ratio_y = h / croped_img.shape[0]

        resize_img = cv2.resize(croped_img, dsize=(w, h), interpolation=cv2.INTER_AREA)
        rh, rw = resize_img.shape[:2]
        img = (np.ones((fh, fw, 3)) * 255).astype(int)
        img[:rh, :rw, :] = resize_img
        
          
        mask = np.zeros((bh, bw))
        #print('mask_shape:',mask.shape)

  
        mask = mask.astype(int) 
        #print('mask_shape:',mask.shape)
        
        # one-hot-encode the mask
        mask = self.ohe_array[mask]
        #print('mask_shape:',mask.shape)

        
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=img, mask=mask)
            img, mask = (sample['image'] / 255).transpose(2,0,1).astype('float32'), sample['mask'].transpose(2,0,1).astype('float32')

       
        #print('image_shape:',img.shape)

        return img, mask
    
    def __len__(self):
        # return length of 
        return len(self.img_path_list)

    
def get_augmentation():
    # Add sufficient padding to ensure image is divisible by 32
    train_transform = [
        album.PadIfNeeded(min_height=448, min_width=640, always_apply=True, border_mode=0),    
    ]
    return album.Compose(train_transform)


def reverse_one_hot(mask, cls_cnt):
    roh = np.arange(1,cls_cnt + 1).reshape(1,1,cls_cnt) * mask
    output = roh.sum(axis=-1).astype(int)
    return output


def process_image(image,rslt_folder_path,img_fnm):
    gray1 = image*100
    gray2 = cv2.equalizeHist(gray1)
    #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) 
    #gray2 = clahe.apply(image)
    blurred = cv2.GaussianBlur(gray2, (5, 5), 0)
    sharpening_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(blurred, -1, sharpening_kernel) * 100

    edges = cv2.Canny(sharpened, 10, 150)
    #cv2.imwrite(f'{img_fnm}_edges.png', edges )

    # 이미지의 고유한 픽셀 값 구하기
    unique_pixel_values = np.unique(gray1)


   
    region_masks = {}
    #unique_pixel_values = new_boundary_dict.keys()
    second_boundary_dict = defaultdict(list)

    ## pixel value --> region 
    
    rdict = { 192:"다목적공간",
              32 :"엘리베이터홀",
            48:"계단실",
            64:"living_room",
            80:"bed_room",
            96 :"kitchen",
            112 :"entrance",
            128  :"valcony",
            144 :"bath_room",
            208   :"실외기실",
            176  :"dress_room",
            192   :"기타",
            208   :"엘리베이터"}

    for pixel_value in unique_pixel_values:
        if pixel_value == 0:
            continue

        mask = (gray1 == pixel_value).astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for region_id in range(1, num_labels):
            region_mask = (labels == region_id).astype(np.uint8) * 255
            region_masks[(pixel_value, region_id)] = region_mask
            output_image = np.zeros_like(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR))
            output_image[region_mask > 0] = [0, 255, 0]

            contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                epsilon = 0.008 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                for point in approx:
                    x, y = point[0]
                    cv2.circle(output_image, (x, y), 5, (255, 0, 0), -1)

                cv2.polylines(output_image, [approx], True, (0, 0, 255), 2)
                #second_boundary_dict[f'region_{pixel_value}_{region_id}']=[point[0].tolist() for point in approx]
                second_boundary_dict[f'{rdict[pixel_value]}_{region_id}']=[point[0].tolist() for point in approx]

                
            #cv2.imwrite(f"{rslt_folder_path}/{img_fnm}second_region_pixel_{pixel_value}_region_{region_id}.png", output_image)

    formatted_list = []
    
    for key, value in second_boundary_dict.items():
        formatted_list.append({
            "name": key,
            "vertex": value
        })

    json_string = json.dumps(formatted_list, ensure_ascii=False, indent=4)

    with open(f'{rslt_folder_path}/{img_fnm}.json', 'w', encoding='utf-8') as json_file:
        json_file.write(json_string)
    '''
    json_string = json.dumps(second_boundary_dict, ensure_ascii=False, indent=4)
           
    # JSON 문자열을 파일로 저장
    with open(f'{rslt_folder_path}/{img_fnm}', 'w', encoding='utf-8') as json_file:
        json_file.write(json_string)
    '''
    #print(f"총 {len(region_masks)}개의 영역이 분리되었습니다.")
    return json_string


    

def evaluate_loop(test_dataset, test_img_path_list, best_model, cls_cnt, DEVICE,rslt_folder_path):
    miou_list = []
    #for idx in range(len(test_dataset)):
    for idx in range(len(test_dataset)):

        img_fnm = test_img_path_list[idx].stem
        image, gt_mask = test_dataset[idx]
        gt_mask_roh = reverse_one_hot(gt_mask.transpose(1,2,0), cls_cnt)

        x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)

        pred_mask = best_model(x_tensor)
        pred_mask = pred_mask.detach().squeeze().cpu().numpy()
        pred_vis = np.max(pred_mask.transpose(1,2,0), axis=-1).round() * (np.argmax(pred_mask.transpose(1,2,0), axis=-1) + 1).astype(int)

        # 원래 사이즈로 늘리기
        bw, bh = 620, 436
        resize_mask = cv2.resize(pred_vis[:bh, :bw].astype(np.uint8), (bw*8, bh*8), interpolation=cv2.INTER_NEAREST)
        rh, rw = resize_mask.shape
        pred_vis_full = np.zeros_like(gt_mask_roh)
        #print('pred_vis_full:',pred_vis_full.shape)

        pred_vis_full[:rh,:rw] = resize_mask
        #print('pred_vis_full:',pred_vis_full.shape)

        
  

        # 예측결과 저장
        pred_vis_full = pred_vis_full.astype(np.uint8)*100
        colored_pred_vis = cv2.applyColorMap(pred_vis_full, cv2.COLORMAP_JET)
        #cv2.imwrite(f'rslt/{img_fnm}_pred.png', pred_vis_full.astype(np.uint8))
        cv2.imwrite(f'rslt/{img_fnm}_pred_colormap.png', colored_pred_vis.astype(np.uint8))

        result = process_image(pred_vis_full,rslt_folder_path,img_fnm)
        return result


        

    


def get_result( cls_cnt, DEVICE, test_model_path, image_path,rslt_folder_path):
  
    #test_img_path_list = sorted((spt / 'images/SPA').glob(f'*_{mtype}_*.PNG'))
    #test_img_path_list = sorted(spt.glob(f'*_{mtype}_*.PNG'))
    test_img_path_list = [image_path]


    test_model = torch.load(test_model_path, map_location=DEVICE).to(DEVICE)

    #check
    print(test_img_path_list)
    

    test_dataset = FloorPlanTestDataset(
        test_img_path_list, 
        cls_cnt=cls_cnt, 
        augmentation=get_augmentation()
    )
    
    #logger.info(f'[{task_nm}/{mtype}] Inference Start')
    point_list = evaluate_loop(test_dataset, test_img_path_list, test_model, cls_cnt, DEVICE,rslt_folder_path)
    #logger.info(f'[{task_nm}/{mtype}] Inference Finished')


    #print(point_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='설명입니다.')
    #parser.add_argument('--fp-model-path', '-fmp', default='./model/SPA_FP_best_model.pth', help='FP 모델 경로를 입력해주세요. 예) ./model/SPA_FP_best_model.pth')
    #parser.add_argument('--cs-model-path', '-cmp', default='./model/SPA_CS_best_model.pth', help='CS 모델 경로를 입력해주세요. 예) ./model/SPA_CS_best_model.pth')
    parser.add_argument('--fp-model-path', '-fmp', default='/c/Users/USER/handy_home/model_custom/model_pram/SPA/SPA_FP_test_model.pth', help='FP 모델 경로를 입력해주세요. 예) ./model/SPA_FP_best_model.pth')
    parser.add_argument('--test_data_path', '-dt', default='/c/Users/USER/handy_home/model_custom/floor_plan/data/image_FP_test.png', help='예측할 데이터의 경로를 입력해주세요. 예) ./model/SPA_CS_best_model.pth')
    parser.add_argument('--result_image_path', '-rt', default='./rslt', help='예측한 결과의 이미지를 저장할 폴더의 경로를 입력해주세요. 예) ./model/SPA_CS_best_model.pth')
    
    args = parser.parse_args()
    
    task_nm = "SPA"
    cls_cnt = 13
    
    fp_test_model_path = Path(args.fp_model_path)
    image_path = Path(args.test_data_path)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    rslt_folder_path = Path(args.result_image_path)
    if not rslt_folder_path.is_dir():
        rslt_folder_path.mkdir()

    get_result(cls_cnt, DEVICE, fp_test_model_path,image_path,rslt_folder_path)
