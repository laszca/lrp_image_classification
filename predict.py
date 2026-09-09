import model
import pickle 

def main():
    # load model
    md = model.load_model()
    
    # preprocess images
    preprocessed_data = model.preprocess_image()
    
    # perform prediction and LRP
    data = model.predict_and_lrp(md, preprocessed_data)

    # save the results to a pickle file
    with open("predictions.pkl", "wb") as f:
        pickle.dump(data, f)
        
if __name__ == "__main__":
    main()
