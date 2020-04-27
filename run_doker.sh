. ./.env;
sudo docker run -p ${HOST_PORT}:5000 --rm -it --env-file .env -v ${INPUT_IMGS_PATH}:/cv_potholes_detection/frames_in -v ${OUTPUT_IMGS_PATH}:/cv_potholes_detection/frames_out docker-cv-pothole-detect;
