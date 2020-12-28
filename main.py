from line_segmenter import line_segment
import os
import shutil
import argparse
from character_segmentor import ConnectedComponents, template_matching
from character_recognition import recognise_hebrew_chars
from style_classification.styleClassificationTest import styleClassification

if __name__ == "__main__":
    arg = argparse.ArgumentParser()
    arg.add_argument('--image', metavar='I', type=str,
                     help="enter the image path", required=True)
    img_parase = arg.parse_args()
    img_path = img_parase.image
    curr_dir = os.getcwd()
    
    # preprocessing 
    if not "lines" in os.listdir():
        os.mkdir("lines")
    if "lines" in os.listdir():
        shutil.rmtree("lines")
        os.mkdir("lines")
    if not "segmented_characters" in os.listdir():
        os.mkdir("segmented_characters")
    if "segmented_characters" in os.listdir():
        shutil.rmtree("segmented_characters")
        os.mkdir("segmented_characters")    
    recognise_hebrew_chars.delete_previous_output()
    
    # main 
    for img in os.listdir(img_path):
        split1 = img.split('.')
        # ----------------------------- line segmentation ---------------------------------------------------
        print(f"------------------------------>> Line segmentation <<----------------------------------------")
        line_segment.run(img_path+"/"+img, "lines/"+split1[0], curr_dir)
        print(f"# Line segmentation completed for image :{str(img)}#")
        os.chdir(curr_dir)
        
        # ----------------------------- character segmentation ----------------------------------------------
        print(f"------------------------------>> character segmentation <<-----------------------------------")
        ConnectedComponents.iterate_over_folders(os.path.join(
            os.getcwd(), 'lines', split1[0]), os.path.join(os.getcwd(), 'segmented_characters'), split1[0])
        template_matching.iterate_over_characters(os.path.join(
            os.getcwd(), 'segmented_characters', split1[0]), os.path.join(os.getcwd(), 'character_images/Images/'))
        print(f"# Completed Character Segmentation for image :{str(img)}")

        # -------------------------- character Recognition -------------------------------------------------
        print(f"------------------------------>> character recognition <<------------------------------------")
        recognise_hebrew_chars.charRecog_main(split1[0])
        print(f"# Completed Character Recognition for image :{str(img)}")
        
        # ------------------------------>> style classification <<------------------------------------------
        print(f"------------------------------>> style classification <<-------------------------------------")
        styleClassification(path_to_char_recog_model="./models/HR_char_recognition.h5", path_to_segmented_images=curr_dir+"/segmented_characters", document_name=split1[0])
        print(f"# Completed Style Classification for image :{str(img)}")
        
