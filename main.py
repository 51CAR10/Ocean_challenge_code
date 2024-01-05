from lib.engine import download_data,eztraction,imageDataGeneration,vgg16_model,base_model,Denselayers,model_train,img_to_tensor_,predict

download_data("https://drive.google.com/file/d/1d6PxFpCPbAQJPuo7yz5dCohUxXQ1rZvR/view?usp=sharing","nasa_images.zip")
eztraction('./','nasa_imagaes.zip')
