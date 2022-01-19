
# Image Caption

Get a subset of the dataset, use a pretrained network to encode the image (ResNet50),
learn the language generator (LSTM). Preferred: include attention mechanisms.


• Paper on a popular approach to Image Captioning: https://arxiv.org/pdf/1411.4555v2.pdf

## FAQ

#### Question 1

Answer 1

#### Question 2

Answer 2


## Documentation

[Documentation](https://linktodocumentation)


## Dataset Format

#### The way on how the Dataset is done follow the way proposed by this dataset Flickr Image Dataset: https://www.kaggle.com/hsankesara/flickr-image-dataset

```
  Images
```
Images need to be in jpeg format, must have a name without space, pre-processed images are better.


```
  Caption
```
The collection of captions need to be placed in a csv file.
**Since** a caption could contain the symbol comma (",") , the separator of each column will be a pipe ("|").
The first row of the file is the header, the column associated are defined in the table below:
| Parameter | Type     | Description                       |
| :-------- | :------- | :-------------------------------- |
| `image_name`      | `string` | File name of the associated caption |
| `comment_number`      | `int` | Index of the caption |
| `comment`*      | `string` | The caption |

*The caption, for a better tokenization, preferably has to be each word separated by a white_space character.
Moreover the simbol dot (".") define the end of the caption.



## Installation

Install my-project with npm

```bash
  npm install my-project
  cd my-project
```
    
## Roadmap

- Additional browser support

- Add more integrations


## Usage/Examples

```javascript
import Component from 'my-project'

function App() {
  return <Component />
}
```


## Running Tests

To run tests, run the following command

```bash
  npm run test
```


## Authors

- [@christiandimaio](https://www.github.com/christiandimaio)

