from pydantic import BaseModel, conlist, validator

# Define the schema for the entire image
class ImageSchema(BaseModel):
    # Each image is a list of lists of lists of floats (224 rows of 224 pixels, each with 3 float values)
    image: conlist(conlist(conlist(float, min_items=3, max_items=3), min_items=224, max_items=224), min_items=224, max_items=224)

    # Add a custom validation to check the range of the pixel values (0-1 for normalized or 0-255 for standard RGB)
    @validator('image', each_item=True)
    def check_pixel_values(cls, pixel_row):
        for pixel in pixel_row:
            if not all(0 <= channel <= 1 for channel in pixel):  # Adjust the range if not using normalized values
                raise ValueError(f'Pixel values must be between 0 and 1. Got: {pixel}')
        return pixel_row

# Define the schema for the array of images
class ImagesSchema(BaseModel):
    images: conlist(ImageSchema, min_items=1)
