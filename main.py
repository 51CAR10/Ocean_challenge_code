from lib.engine import download_data,eztraction,train_Generation,test_Generation,val_Generation,DataGeneration,vgg16_model,base_model,Denselayers,model_train,img_to_tensor_,predict


def main():
    download_data("https://drive.google.com/file/d/1d6PxFpCPbAQJPuo7yz5dCohUxXQ1rZvR/view?usp=sharing","nasa_images.zip")
    eztraction('./','nasa_imagaes.zip')
    train_Generation('./nasa_images/',0.2,16,["debris","no_debris"],150)
    val_Generation('./nasa_images/',0.2,16,["debris","no_debris"],150)
    vgg_model=vgg16_model((150,150),"imagenet")
    model=base_model(vgg_model)
    model=Denselayers(model)

    model_train(model,train_generator=train_Generation(),val_generator=val_Generation(),checkpoint_path='ocean_challenge_fine_tuning_vgg16.h5')
if __name__=="__main__":
    main()