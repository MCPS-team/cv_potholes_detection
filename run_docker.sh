. ./.env;
sudo docker run -p ${HOST_PORT}:5000 --rm -it \
    --env-file .env \
    -v ${INPUT_IMGS_DIR}:/cv_potholes_detection/frames_in \
    -v ${OUTPUT_IMGS_DIR}:/cv_potholes_detection/frames_out \
    -v ${MODEL_DIR}:/cv_potholes_detection/models \
    docker-cv-pothole-detect;
