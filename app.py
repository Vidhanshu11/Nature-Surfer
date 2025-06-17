from PIL import Image
import gradio as gr
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import os 


# formatting functions
def format_underscores(input_string):
    '''
    Formats an input string by capitalizing each word separated by underscores.

    Parameters:
        - input_string (str): The input string containing underscores.

    Returns:
        str: The formatted string with words capitalized and underscores replaced by spaces.

    Example usage:
        formatted_string = format_underscores("hello_world")
        print(formatted_string)  # Output: "Hello World"
    '''
    words = input_string.split('_')
    formatted_words = [word.capitalize() for word in words]
    formatted_string = ' '.join(formatted_words)
    return formatted_string

def list_to_string(input_list, commas=False):
    '''
    Converts a list of items into a formatted string.

    Parameters:
        - input_list (list): The list of items to be converted to a string.
        - commas (bool): If True, separate items with commas; otherwise, separate with spaces.

    Returns:
        str: The formatted string containing items from the list.

    Example usage:
        items = ['apple', 'banana', 'orange']
        result_string = list_to_string(items, commas=True)
        print(result_string)  # Output: "apple, banana, orange"
    '''
    # Convert each item in the list to a string
    str_list = [format_underscores(str(item)) for item in input_list]
    # Join the items with spaces between them
    if commas:
        result_string = ', '.join(str_list)
    else:
        result_string = " ".join(str_list)
    return result_string

def return_disease_plant_name(string):
    """
    Extracts the name and disease information from a label string in the format '<name>__<disease>_<disease>'.

    Parameters:
    - string (str): A string containing information in the format '<name>__<disease>_<disease>'.

    Returns:
    - tuple: A tuple containing two elements:
        - name (str): The name extracted from the input string.
        - disease (str): The disease information extracted from the input string, with multiple underscores formatted.

    Example:
    If the input string is 'plant_name__leaf_spot_disease', this function will return ('plant_name', 'leaf spot disease').

    Note:
    The function assumes that the input string follows the specified format, with double underscores separating the name and disease information.

    """
    first_underscore_index = string.find('_')
    name = format_underscores(string[0:first_underscore_index])
    disease = format_underscores(string[first_underscore_index + 2:])
    return (name, disease)

#unique_diseaases intialisation
unique_diseases = np.array(['Apple___Apple_scab', 'Apple___Black_rot',
        'Apple___Cedar_apple_rust', 'Apple___healthy',
        'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
        'Cherry_(including_sour)___healthy',
        'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
        'Corn_(maize)___Common_rust_',
        'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
        'Grape___Black_rot', 'Grape___Esca_(Black_Measles)',
        'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
        'Orange___Haunglongbing_(Citrus_greening)',
        'Peach___Bacterial_spot', 'Peach___healthy',
        'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
        'Potato___Early_blight', 'Potato___Late_blight',
        'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
        'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch',
        'Strawberry___healthy', 'Tomato___Bacterial_spot',
        'Tomato___Early_blight', 'Tomato___Late_blight',
        'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
        'Tomato___Spider_mites Two-spotted_spider_mite',
        'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
        'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'])

#model loading
def load_model(model_path):
  """
  Loads a saved model from a specified path.
  """
  model = tf.keras.models.load_model(model_path,
                                     custom_objects={"KerasLayer":hub.KerasLayer})
  print('Successful')
  return model
model = load_model(model_path = "-complete-model-2-epochs.h5")

#image processing and data batching 
def process_image(image_path):
    """
    Preprocesses an image file.

    Args:
        image_path (str): The file path of the image.

    Returns:
        tf.Tensor: Preprocessed image tensor.

    Preprocesses an image file by reading the image from the specified file path, decoding it as a JPEG image,
    normalizing the pixel values, and resizing the image to a target size of (224, 224) pixels.

    Returns the preprocessed image as a tensor.

    """
    # read in the image file from the filepath
    image = tf.io.read_file(image_path)
    # decode the image to a jpeg
    image = tf.image.decode_jpeg(image, channels=3)
    # image normalization
    image = tf.image.convert_image_dtype(image, tf.float32)
    # resize the image
    image = tf.image.resize(image, size=(224, 224))

    return image

def get_data_batches(X, y = None, batch_size = 32, valid_data = False, test_data = False, deployment = False):
  """
    Create batches of data for training, validation, or testing.

    Parameters:
        X (numpy array): Input data (image file paths) as a NumPy array.
        y (numpy array, optional {not given when creating batched for test data}): Labels for the input data. Default is None.
        batch_size (int, optional): Size of each batch. Default is 32.
        valid_data (bool, optional): If True, create batches for validation data. Default is False.
        test_data (bool, optional): If True, create batches for test data. Default is False.
        deployment (bool, optional): If True, don't show download symbol. Default is False.

    Returns:
        tf.data.Dataset: A TensorFlow Dataset containing the data in batches.

    Note:
        - If `test_data` is True, the function creates batches for test data without labels.
        - If `valid_data` is True, the function creates batches for validation data with labels.
        - If both `valid_data` and `test_data` are False, the function creates batches for training data with labels.
        - The input data (X) and labels (y) are converted to TensorFlow constant tensors.
        - The `process_image` and `get_image_label` functions are used to process the images and labels.
        - For training data, the dataset is shuffled before creating batches. This is in a bid to remove bias
    """
  if test_data:
    if deployment:
      data = tf.data.Dataset.from_tensor_slices(tf.constant(X))
      data_batch = data.map(process_image).batch(batch_size)
  return data_batch


# model pipeline
def complete_model_pipeline(image):
    """
    Complete model pipeline for deploying the model using Gradio.

    This function takes an image as input, saves it temporarily to a file,
    loads the image as a test sample, performs prediction using the model,
    and then removes the temporarily saved image. The function returns the
    predicted label

    Args:
        image (np.ndarray): The NumPy array representing the input image.

    Returns:
        str: The predicted label.

    Parameters:
        - image: A NumPy array representing the input image.

    Note: This function is designed to be used with Gradio to deploy the model
    and obtain predictions interactively.

    """

    # Define the file path to temporarily save the image



    image_file_path = 'test-image.jpg'

    # Convert the input NumPy array to a Pillow image object and save it as a file
    image = Image.fromarray(image)
    image.save(image_file_path)

    # Prepare test data for the model using the saved image file
    test_data = get_data_batches(X=[image_file_path], test_data=True, batch_size=1, deployment = True)

    # Perform prediction using the model
    predictions = model.predict(test_data, verbose = 0)

    # Get the index of the highest predicted value and obtain the corresponding breed label
    predicted_label_index = np.argmax(predictions[0])
    predicted_disease_label = unique_diseases[predicted_label_index]
    prediction_probability = np.max(predictions[0])
    
    # Remove the temporarily saved image file
    os.remove(image_file_path)
    result = return_disease_plant_name(predicted_disease_label)
    border = '-' * 40  # Adjust the number of hyphens as needed
    if prediction_probability > 90:
        formatted_string = f'''
    +----------------------------------------+
    |  Name of your plant: {result[0]:<20}
    |  Disease your plant is facing: {result[1]:<20}
    |  Prediction Probability: {prediction_probability*100:.2f}%
    +----------------------------------------+
    '''
    else:
       formatted_string = f'''
    +----------------------------------------+
    |  The prediction probability is too low for the output to be produced
    +----------------------------------------+
    '''
    return formatted_string





iface = gr.Interface(
    fn=complete_model_pipeline,
    inputs="image",
    outputs="text",
    title = "Plant Disease Classification Application",
    description = 'Upload the image of a plant with a disease and get to know which disease it is facing (or not facing :))'
)

iface.launch()