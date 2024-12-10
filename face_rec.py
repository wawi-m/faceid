# Iterate through the dataset and load images
for person_id in os.listdir(image_dir):
    person_path = os.path.join(image_dir, person_id)
    
    if os.path.isdir(person_path):
        for image_name in os.listdir(person_path):
            image_path = os.path.join(person_path, image_name)
            image = cv2.imread(image_path)
