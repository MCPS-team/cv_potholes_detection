. ./.env;
sudo docker run -p ${HOST_PORT}:5000 --rm -it --env-file .env -v /var/www/MCPS/cv_potholes_detection/:/cv_potholes_detection2 docker-cv-pothole-detect bash 
