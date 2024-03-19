from tools import csv2dictlist, normalize
import numpy as np
from matplotlib import cm
import cv2
from generate_heatmaps import create_heatmap
from dicom_imgs import DicomImgs


class ReflacxSample:
    def __init__(self, dicom_id, reflacx_id, sample_dict):
        self.data = sample_dict
        self.dicom_id = dicom_id
        self.reflacx_id = reflacx_id
        self.dicom_img = None
        self.chest_bb = None
        self.fixations = None
        self.timed_sentences = None
        self.global_heatmap = None
        self.heatmaps_by_sentence = None
        self.anomaly_ellipses = None

        self.color_gen = lambda cmap: lambda ratio: tuple((int(255 * comp) for comp in cmap(ratio)[:3]))


    def canvas(self):
        canvas = np.copy(self.get_dicom_img())
        canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2RGB)
        canvas >>= 4
        return canvas

    
    def get_dicom_img(self):
        return DicomImgs.get_dicom_img(self.dicom_id, imgpath=self.data['image'])
    

    def get_chest_bounding_box(self):
        if self.chest_bb is None:
            self.chest_bb = csv2dictlist(self.data['chest_bounding_box'])[0]
            dicom_img = self.get_dicom_img()
            self.chest_bb['xmin'] = min(dicom_img.shape[0],
                                        max(self.chest_bb['xmin'], 0))
            self.chest_bb['xmax'] = min(dicom_img.shape[0],
                                        max(self.chest_bb['xmax'], 0))
            self.chest_bb['ymin'] = min(dicom_img.shape[1],
                                        max(self.chest_bb['ymin'], 0))
            self.chest_bb['ymax'] = min(dicom_img.shape[1],
                                        max(self.chest_bb['ymax'], 0))
            
            
        return self.chest_bb
    
    
    def get_cropped_chest_img(self):
        bb = self.get_chest_bounding_box()
        return np.copy(self.get_dicom_img()[bb['ymin']: bb['ymax'],
                                            bb['xmin']: bb['xmax']])
    

    def get_fixations(self):
        if self.fixations is None:
            self.fixations = csv2dictlist(self.data['fixations'])
        return self.fixations
    

    def draw_fixations(self, cmap='jet'):
        fixations = self.get_fixations()
        delta_t = fixations[0]['timestamp_start_fixation'], fixations[-1]['timestamp_end_fixation']
        f_color = self.color_gen(cm.get_cmap(cmap))

        canvas = self.canvas()

        for fixation in fixations:
            x = int(fixation['x_position'])
            y = int(fixation['y_position'])
            
            if x < 0 or y < 0:
                continue

            median_t = (fixation['timestamp_end_fixation'] + fixation['timestamp_start_fixation']) / 2
            rel_t = (median_t - delta_t[0]) / (delta_t[1] - delta_t[0])

            canvas = cv2.circle(canvas, (x, y), 40, f_color(rel_t), -1)

        return canvas
    

    def get_timed_sentences(self):
        if self.timed_sentences is None:
            with open(self.data['transcription']) as f:
                sentences = [sentence.strip(' \n')
                             for sentence in ''.join(f.readlines()).split('.')
                             if sentence != '']
                
            tt = csv2dictlist(self.data['timestamps_transcription'])

            start_t = 0
            end_t = 0

            timed_sentences = []

            new_sentence = False
            for i, token in enumerate(tt):
                if i == 0 or new_sentence:
                    start_t = token['timestamp_start_word']
                    end_t = token['timestamp_end_word']
                    new_sentence = False

                if token['word'] == '.':
                    timed_sentences.append({'start_t': start_t,
                                                 'end_t': end_t,
                                                 'sentence': sentences.pop(0)})
                    new_sentence = True

                end_t = token['timestamp_end_word']

            sentence_i = 0
            sentence = timed_sentences[sentence_i]

            fixations = self.get_fixations()

            pre_transcript = []
            post_transcript = []

            for fixation in fixations:
                x = fixation['x_position']
                y = fixation['y_position']
                
                if x < 0 or y < 0:
                    continue

                if (sentence_i == 0
                    and fixation['timestamp_start_fixation'] < sentence['start_t']):
                    pre_transcript.append((fixation['timestamp_start_fixation'], fixation))
                else:
                    while fixation['timestamp_end_fixation'] > sentence['end_t']:
                        if sentence_i >= len(timed_sentences) - 1:
                            post_transcript.append((fixation['timestamp_end_fixation'], fixation))
                            break
                        else:
                            sentence_i += 1
                            sentence = timed_sentences[sentence_i]
                    if 'fixations' not in sentence:
                        sentence['fixations'] = []
                    sentence['fixations'].append(fixation)

            if len(pre_transcript) > 0:
                timed_sentences.insert(0, {'start_t': pre_transcript[0][1]['timestamp_start_fixation'],
                                           'end_t': pre_transcript[-1][1]['timestamp_end_fixation'],
                                           'sentence': '_pre_transcript',
                                           'fixations': [f[1] for f in pre_transcript]})
                
            if len(post_transcript) > 0:
                timed_sentences.append({'start_t': post_transcript[0][1]['timestamp_start_fixation'],
                                        'end_t': post_transcript[-1][1]['timestamp_end_fixation'],
                                        'sentence': '_post_transcript',
                                        'fixations': [f[1] for f in post_transcript]})
            
            self.timed_sentences = timed_sentences
        
        return self.timed_sentences
    

    def draw_fixations_by_sentence(self, cmap='jet', radius=40):
        timed_sentences = self.get_timed_sentences()
        f_color = self.color_gen(cm.get_cmap(cmap))

        result = {}

        for sentence in timed_sentences:
            canvas = self.canvas()
            
            for i, fixation in enumerate(sentence['fixations']):
                x = int(fixation['x_position'])
                y = int(fixation['y_position'])

                canvas = cv2.circle(canvas,
                                    (x, y),
                                    radius,
                                    f_color(i / max(1, (len(sentence['fixations']) - 1))),
                                    -1)

                result[sentence['sentence']] = canvas
        
        return result


    def get_heatmap(self, chest_only=False):
        if self.global_heatmap is None:
            hm = np.load(self.data['heatmaps'], allow_pickle=True).item()['np_image']
            self.global_heatmap = normalize(hm, type=hm.dtype)
        
        if not chest_only:
            return np.copy(self.global_heatmap)
        
        bb = self.get_chest_bounding_box()
        result = np.copy(self.global_heatmap[bb['ymin']: bb['ymax'],
                                           bb['xmin']: bb['xmax']])
        return result / np.sum(result)
    

    def get_heatmaps_by_sentence(self, chest_only=False):
        if self.heatmaps_by_sentence is None:
            timed_sentences = self.get_timed_sentences()
            get_partial_hm = lambda fixations: create_heatmap(fixations,
                                                    self.data['image_size_x'],
                                                    self.data['image_size_y'])
            
            hms = []

            for sentence in timed_sentences:
                img = get_partial_hm(sentence['fixations'])

                if chest_only:
                    bb = self.get_chest_bounding_box()
                    img = img[bb['ymin']: bb['ymax'], bb['xmin']: bb['xmax']]
                    img /= np.sum(img)
                
                hms.append({'title': sentence['sentence'],
                            'img': img,
                            'start_t': sentence['fixations'][0]['timestamp_start_fixation'],
                            'end_t': sentence['fixations'][-1]['timestamp_end_fixation']})
            
            self.heatmaps_by_sentence = hms
            
        return self.heatmaps_by_sentence
    

    def get_anomaly_ellipses(self):
        if self.anomaly_ellipses is None:
            self.anomaly_ellipses = csv2dictlist(self.data['anomaly_location_ellipses'])
        return self.anomaly_ellipses
    

    def draw_anomaly_ellipses(self, color = (255, 0, 0), chest_only=False):
        ellips = self.get_anomaly_ellipses()

        result = {}

        for ellip in ellips:
            x_min = ellip['xmin']
            x_max = ellip['xmax']
            y_min = ellip['ymin']
            y_max = ellip['ymax']

            contour = cv2.fitEllipse(np.array([(x_min, y_min), (x_min, y_max), (x_max, y_max), (x_max, y_min), (x_min, y_min)]))

            canvas = self.canvas()
            canvas = cv2.ellipse(canvas, contour, color, 15)
            if chest_only:
                bb = self.get_chest_bounding_box()
                canvas = canvas[bb['ymin']: bb['ymax'], bb['xmin']: bb['xmax']]
            anomalies = ', '.join([key for key in ellip if str(ellip[key]) == 'True'])
            result[anomalies] = canvas

        return result