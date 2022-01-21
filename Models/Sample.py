from PIL import Image
import os

class Sample():
    """Model class for managing a sigle sample of the dataset.
    """
    
    def __init__(self, id: int, image_path_file: str, caption: str, verbose: bool = False):
        """Constructor of a sample of the dataset.

        Args:
            id (int): The id associated with this sample.
                        [Constraint: The id must be unique (Caller responsibility).]
                        [Constraint: The id must be greater than 0.]
            image_path_file (str): The image path, in relative path format. (Assumed the main.py as the entry point)
            caption (str): Raw caption associated with this sample. 
                        [Constraint: Caption has to be a string with length greater than 1 characters.]
            verbose (bool, optional): The class will be verbose if True. Defaults to False.
        Raises:
            FileNotFoundError: The given relative path to the image is invalid.
            ValueError: The caption is invalid.
            ValueError: The id is invalid.
        """
         
        # Validation of constructor parameters
        if not os.path.isfile(image_path_file):
            raise FileNotFoundError("The given path_file resemble a non-existing file.")
        
        if len(caption.strip()) <= 1:
            raise ValueError(f"The caption has a length of {len(caption.strip())} characters, which is not supported.")
        
        if id <= 0:
            raise ValueError(f"The id must be greater than 0. \n Given {id}.")
        
        if verbose:
            print(f"Image path: {image_path_file}")
            print(f"Loading..")
        
        # Loading the image
        self._image = None
        try:
            self._image = Image.open(image_path_file).convert('RGB') # Load and convert to RGB
        except Exception as e:
            raise e
        if self._image is None:
            raise Exception("Could not load image.")
        if verbose:
            print("Ok.")
        
        # Loading the caption 
        self.caption = caption
        
        # Loading the id
        self.id = id
        
        # Set verbosity 
        self._verbose = verbose
        
        # Tell externally if this sample is altered (pre-processed, or other..) if False, otherwise the data is inside are raw if True
        self.is_raw = True
        
    @property
    def image(self) -> Image:
        """Getter of the image property

        Returns:
            Image: The image object.
        """
        print("Getter called.")
        return self._image
    
    def alter_image(self, altered_image: Image):
        """Alter the sample image by place a new one (could be the same but modified or another one) 

        Args:
            image (Image): The new image
        """
        self.image = altered_image
        self.is_raw = False
        
    def alter_caption(self, altered_caption):
        """Alter the caption, now could be a string as before or a list of string, ready for being processed by the NN.

        Args:
            image (Image): The new image
        """
        self.caption = altered_caption
        self.is_raw = False
        
    
        