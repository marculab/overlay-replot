from scipy.io import loadmat
import scipy.sparse.csgraph._validation
import numpy as np
from PIL import Image, ImageDraw
import cv2
import csv
import sys

def GetColor(val, scale_from, scale_to, dest_channel):
    if val < scale_from[dest_channel - 1]:
        val = scale_from[dest_channel - 1]
    elif val > scale_to[dest_channel - 1]:
        val = scale_to[dest_channel - 1]
    scaled_val = int((val - scale_from[dest_channel - 1]) / (scale_to[dest_channel - 1] - scale_from[dest_channel - 1]) * 255)
    color = cv2.applyColorMap(np.array(scaled_val, dtype='uint8'),  cv2.COLORMAP_JET)
    c = np.array([color[0][0][0], color[0][0][1], color[0][0][2]], dtype='float')
    return c

def main(argv):
    ## 3 or 4 arguments
    if len(argv) != 3 and len(argv) != 4:
        print("usage: replot <dataMatFileName> <videoFileName> (optional)<motionCorrectionMatFileName>")
        return
    
    datafilename = argv[1]
    videofilename = argv[2]
    if len(argv) == 4:
        motion_correct = True
        posfilename = argv[3]
    else:
        motion_correct = False

    outputfile = videofilename.split('.')[0] + "_replot.avi"

    ## plotConfig File
    configfilename = "plotConfig.csv"
    try: 
        with open(configfilename, newline='') as csvfile:
            params = list(csv.reader(csvfile, delimiter=','))
            dest_channel = int(params[0][1])
            plot_intensity = int(params[1][1])
            snr_lowerbound = float(params[2][1])
            radius = int(params[3][1])
            alpha = float(params[4][1])
            lt_scale_from = [float(params[5][1]), float(params[7][1]), float(params[9][1])]
            lt_scale_to = [float(params[6][1]), float(params[8][1]), float(params[10][1])]
            int_scale_from = [float(params[11][1]), float(params[13][1]), float(params[15][1])]
            int_scale_to = [float(params[12][1]), float(params[14][1]), float(params[16][1])]
    except IOError:
        print("plotConfig.csv file does not exist, using default values")
        ## use default values
        dest_channel = 2
        alpha = 0.5
        radius = 25
        plot_intensity = 0
        lt_scale_from = [3.5, 3, 2.5]
        lt_scale_to = [5, 6, 5]

    ## load matfile into arrays
    # datafilename = "Patient_23_Run_P11.mat"

    datamat = loadmat(datafilename)
    print("Data mat loaded...")
    data_attr_key = list(datamat.keys())[3] ## for exp. ['__header', '__version', '__globals__', 'Patient_23_Run_P11']
    lt_data = datamat[data_attr_key][:, 11:15] ## (N, 4) array, N is number of data points
    int_data = datamat[data_attr_key][:, 15:19]
    snr_data = datamat[data_attr_key][:, 19:23  ]
    xy_data = datamat[data_attr_key][:, 6:8]

    ## load video
    # videofilename = "P528731_R01_P23_run11.avi"
    vc = cv2.VideoCapture(videofilename)
    print("Video file loaded...")
    fps = vc.get(cv2.CAP_PROP_FPS)
    suc, image = vc.read()
    frame_idx = 0

    ## load motion correction position file
    # posfilename = "PosDataMC.mat"
    if motion_correct:
        mc_posmat = loadmat(posfilename)
        pos_attr_key = list(mc_posmat.keys())[3]
        ## mc_posmat[pos_attr_key][N][0].shape = (N, 2)
        print("Motion correction position file loaded...")

    ## coordinates originally stored in (1280, 720) grid; video's frame size might be different
    v_height = image.shape[0]
    v_width = image.shape[1]

    ## initialize video writer
    # outputfile = "out.avi"
    vw = cv2.VideoWriter(outputfile, cv2.VideoWriter_fourcc(*'FMP4'), fps, (v_width, v_height))
    color_dict = {}

    '''
    //threadholding mask, technically it saves time by reduced computation. but mask calculation seems bo be very time comsuming..
    overlay = np.ones((v_height, v_width, 3), dtype='uint8')
    ret, mask = cv2.threshold(overlay[:,:,0], 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
    imcp = image.copy()
    imcp[np.where(mask == 255)] = overlay[np.where(mask == 255)]
    vw.write(cv2.addWeighted(imcp, alpha, image, 1 - alpha, 0))
    '''


    ## replot
    if not plot_intensity:
        print("--- Plotting Lifetime Channel " + str(dest_channel) + " ---")
    else:
        print("--- Plotting Intensity Channel " + str(dest_channel) + " ---")
    point_number = xy_data.shape[0]
    while suc:
        print("frame: " + str(frame_idx + 1))
        ## in older cases video frame numbers may be more than point numbers
        if frame_idx == point_number:
            break
        if not motion_correct: ## use the x, y coordinates stored in the data mat file
            overlay = image.copy()
            for p_idx in range(frame_idx + 1): ## replot every point seems to be faster than saving the mask...
                position = xy_data[p_idx]
                if p_idx in color_dict:
                    cv2.circle(overlay, (int(position[0] / 1280. * v_width) , int(position[1] / 720. * v_height)), int(radius), color_dict[p_idx], -1)
                else:            
                    if position[0] != 0 or position [1] != 0: ## point exist
                        if not plot_intensity: ## plot lifetime
                            ltval = lt_data[p_idx][dest_channel - 1]
                            snrval = snr_data[p_idx][dest_channel - 1]
                            if not (np.isnan(snrval) or snrval < snr_lowerbound or np.isnan(ltval) or ltval == 0): ## valid point
                                c = GetColor(ltval, lt_scale_from, lt_scale_to, dest_channel)
                                color_dict.setdefault(p_idx, c)
                                cv2.circle(overlay, (int(position[0] / 1280. * v_width) , int(position[1] / 720. * v_height)), int(radius), c, -1)
                        else: ## plot intensity value
                            intval = int_data[p_idx][dest_channel - 1]
                            snrval = snr_data[p_idx][dest_channel - 1]
                            if not (np.isnan(snrval) or snrval < snr_lowerbound or np.isnan(intval) or intval <= 0 or intval > 1): ## valid point
                                c = GetColor(intval, int_scale_from, int_scale_to, dest_channel)
                                color_dict.setdefault(p_idx, c)
                                cv2.circle(overlay, (int(position[0] / 1280. * v_width) , int(position[1] / 720. * v_height)), int(radius), c, -1)
            vw.write(cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0))
            suc, image = vc.read()
            frame_idx += 1
        else:
            overlay = image.copy()
            for p_idx in range(frame_idx + 1):
                position = mc_posmat[pos_attr_key][frame_idx][0][p_idx]
                if p_idx in color_dict:
                    cv2.circle(overlay, (int(position[0] / 1280. * v_width) , int(position[1] / 720. * v_height)), int(radius), color_dict[p_idx], -1)
                else:
                    if position[0] != 0 or position [1] != 0: ## point exist
                        ## check data point's property
                        if not plot_intensity:
                            ltval = lt_data[p_idx][dest_channel - 1]
                            snrval = snr_data[p_idx][dest_channel - 1]
                            if not (np.isnan(snrval) or snrval < snr_lowerbound or np.isnan(ltval) or ltval == 0): ## valid point              
                                c = GetColor(ltval, lt_scale_from, lt_scale_to, dest_channel)
                                color_dict.setdefault(p_idx, c)
                                ## draw circle
                                cv2.circle(overlay, (int(position[0] / 1280. * v_width) , int(position[1] / 720. * v_height)), int(radius), c, -1)
                        else:
                            intval = int_data[p_idx][dest_channel - 1]
                            snrval = snr_data[p_idx][dest_channel - 1]
                            if not (np.isnan(snrval) or snrval < snr_lowerbound or np.isnan(intval) or intval <= 0 or intval > 1): ## valid point
                                c = GetColor(intval, int_scale_from, int_scale_to, dest_channel)
                                color_dict.setdefault(p_idx, c)
                                cv2.circle(overlay, (int(position[0] / 1280. * v_width) , int(position[1] / 720. * v_height)), int(radius), c, -1)

                # cv2.imwrite(str(frame_idx) + '.png', image)
            vw.write(cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0))
            suc, image = vc.read()
            frame_idx += 1


if __name__ == "__main__":
    main(sys.argv)

