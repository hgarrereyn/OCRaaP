# OCRaaP

![Logo](logo.png)

OCRaaP (Optical Character Recognition as a Program) is a handwritten, two-dimensional programming language designed for maximum ease of use and portability (all you need is some paper and a pen).

# Quick usage

You can run the example programs like so:

*While in the root directory,*

`python interpreter.py -i examples/helloWorld.jpg`

This will run the program and print any output. If the program requires input, it will halt and wait for you to enter a string in the terminal.

Alternatively, to debug a program, add the `-d` flag:

`python interpreter.py -i examples/helloWorld.jpg -d`

This will open up a window where you can step through the program. Simply press any key while the window is active to perform one execution step. Additionally, the current symbol, coordinate and stack values will be printed in the terminal.

# In this repo

In this repo you will find the following files and folders:

* `interpreter.py` - A CLI interpreter with a built-in debugger.

  * Run: `python interpreter.py -h` for options

* `model.py` - Represents the neural network classifier and provides an interface to facilitate prediction and training.

* `image_parser.py` - Performs symbol detection on a image.

* `generate.py` - A command line tool to help generate testing and training data for the classifier.

* `data_loader.py` - A tool to load and process images that the classifier will use for training or prediction.

# Overview

An OCRaaP program is first drawn on paper and then scanned in to the computer where it is fed into an interpreter.

The code pointer is a 2D projectile that moves around the paper and triggers operations upon encountering symbols similar.

Execution begins at `sad` and moves towards the nearest symbol. Once it reaches this symbol, it executes the operation and attempts to continue moving in the same direction towards the next nearest symbol within ~20 degrees of its current heading. If there is no nearby symbol, the code pointer will perform a "long jump" where it will attempt to move to the closest symbol that is directly along the current heading.

*Note: the user is advised to make the code path as clear as possible since the interpreter can be quite a pain in the ass*

When the code pointer reaches `Dead`, it halts execution.

Operations are conducted on an operand stack that holds unbounded integers.

# Install Dependencies

You'll need the following dependencies to run OCRaaP:
 * CV2
 * NumPy
 * IMUtils
 * TensorFlow

## Ubuntu (and Debian distributions)

You probably already have git and and pip, but don't forget about OpenCV. IMUtils and TensorFlow require escalated permissions to install.

 > sudo apt-get install git python-pip python-opencv
 > pip install --upgrade pip
 > pip install numpy cv2
 > sudo -H pip install imutils tensorflow

# Examples

The following images are examples of valid OCRaaP programs:

## Hello World 
Prints the string `Hello World!` to the console.
![HelloWorld](examples/helloWorld.jpg)

OCRaap uses a **LIFO** (Last In First Out) stack. Our *Hello World* will put the string "Hello World!" on the stack, and then it will print it off one character at a time.

### Adding to the Stack
We'll start with a binary zero on the stack; `0$`. This will, eventually, signal that our program is finished executing. We proceed to add the [ASCII](https://en.wikipedia.org/wiki/ASCII) encoded characters, in reverse order, on the stack. `100001$0010011$0011011$0100111$1111011$`, and so on.

> The string is going on in reverse, 100001 (!) 0010011 (d) 0011011 (l) 0100111 (r) 1111011 (o), and so on, because the stack is LIFO. The values themselves are Little [Endian](https://en.wikipedia.org/wiki/Endianness), so "d", the fourth letter of the alphabet, is encoded 0010011 (0 + 0 + 4 + 0 + 0 + 32 + 64), not 1100100. The highest bits indicate the case of the letter, 96 for lower case and 64 for upper. This is why "W", the upper case version of the twenty-third letter of the alphabet, is encoded 1110101 (1 + 2 + 4 + 0 + 16 + 0 + 64).

Notice that we have a repeated $ at our pair of "l"s. Rather than write out the sequence 0011011 again, we can simply repeat $ to duplicate the latest stack entry.

Finally, we'll move the execution pointer to the printer. A non-visual language might just `print(var)`, but in OCRaaP we'll need to draw the shape. We put a zero on the stack with `0000$` and then put the sum of the last two constants we entered, a binary zero and `H`, onto the stack with `+$`.

> We can draw without affecting the logic of the program with a **NOP** (No OPeration) Pattern; `O$+$`. Working backwards, we see that we're putting a sum on the stack. Specifically, we're putting on the sum of binary 0 and whatever the most recent stack member is. And that sum happens to be... Whatever the most recent stack member is. We can lengthen it by padding the zero with more zeroes- It still resolves to binary zero.

### Printing from the Stack
Now let's explore our printer. We have `?(E)(#OOOOOOO$+$)` where the left branch ends the program and the right hand branch loops back to our `?`.

We begin by checking if the last stack entry was a binary zero. The most recent entry is `H`, so we take the right hand branch. This prints it with `#`. This pops it off the stack.

Next we'll use our NOP pattern to loop back to the `?`. This loop continues until we find the original binary zero that we started our program with. Since it *is* a binary zero, we'll take the left branch, which ends our program with `E`.

## Echo
Reads a string from the console and then prints it back.
![Echo](examples/echo.jpg)

The simplest version of this program would be `S@$#E`, but that would stop after one character. 

We can add a loop by drawing the shape with a NOP pattern; `S0000000000$@$+$#` and point the `#` back in a loop somewhere in the `0` pad, but this would *never* stop.

We can choose to stop if the character picked up with `@` is a binary zero by introducing a `?`. If we draw `S000000000$@$+$?(E)(#000000000)` with the right hand branch point at our NOP's 0 pad, we will end if the program recieves a binary zero. Our loop looks like a big circle.

> Notice that the NOP is rejoined by weaving the end pad with the start pad. On our first execution we'll have nine drawn zeroes which resolve to binary zero, and on each subsequent loop we'll have *eighteen* drawn zeroes which still resolve to binary zero.

Finally, we can make our code more aesthetically interesting by twisting our circle and reusing one of our drawn zeroes. This doesn't do anything logically, but is interesting spatially.

The example above has aslightly different implementation which uses the same techniques; `S00000$@$?(E)(000000$+#000)`. In this implementation, the stack gets another binary zero added with each round.

# Symbols

### Global

| Symbol | Name | Description |
| --- | --- | --- |
| ![sad](examples/symbols/sad.jpg) | `Sad` | The initial execution point of the program. If there are more than one of these, the program is invalid. |
| ![dead](examples/symbols/dead.jpg) | `Dead` | Code execution stops here |
| ![at](examples/symbols/at.jpg) | `At` | Reads a character from the input stream and pushes it to the stack |
| ![hash](examples/symbols/hash.jpg) | `Hash` | Pops a character from the stack and writes it to the output stream |

### Conditional

| Symbol | Name | Description |
| --- | --- | --- |
| ![conf](examples/symbols/conf.jpg) | `Conf` | Pops a value and checks if it is zero. If it is, code execution is deflected ~60 deg counter clockwise. Otherwise, code execution is deflected ~60 deg clockwise. |

### Constant

Constants can be pushed to the stack in binary notation termiated by a `Dollar`. Operands begin at the least significant bit.

For example, the following path of symbols would push 13 (0b1101) to the stack:

```
Dot -> Empty -> Dot -> Dot -> Dollar
```

| Symbol | Name | Description |
| --- | --- | --- |
| ![empty](examples/symbols/empty.jpg) | `Empty` | Writes a binary 0 |
| ![dot](examples/symbols/dot.jpg) | `Dot` | Writes a binary 1 |
| ![dollar](examples/symbols/dollar.jpg) | `Dollar` | Terminates const loading |

### Arithmetic

| Symbol | Name | Description |
| --- | --- | --- |
| ![plus](examples/symbols/plus.jpg) | `Plus` | pop b, pop a, push a+b |
| ![dash](examples/symbols/dash.jpg) | `Dash` | pop b, pop a, push a-b |

# Errors

### 1x - Parse Time

| Code | Name | Description |
| --- | --- | --- |
| 10 | too sad | Found duplicate `sad` symbol |
| 11 | too happy | Could not find `sad` symbol |

### 2x - Runtime

| Code | Name | Description |
| --- | --- | --- |
| 20 | stack underflow | tried to pop from empty stack |
| 21 | already sad | tried to execute a `sad` symbol after start |

# How it works

### Classifier

The intepreter in this repository uses a convolutional neural network classifer that was trained using TensorFlow on a few dozen of each symbol. Training symbols were generated in every orientation so the classifier should be rotation invariant. It reached about `97%` accuracy on a separate validation set.

*Note: users are advised to draw symbols with great precision so as not to confuse the classifier*

The network structure is as follows:

```
INPUT: 40x40x1 grayscale images

Convolutional Layer 1: 40x40x6 (kernel: 6x6, stride: 1)
Convolutional Layer 2: 20x20x12 (kernel 5x5, stride: 2)
Convolutional Layer 3: 10x10x24 (kernel 4x4, stride: 2)

Fully Connected + dropout: 200

Softmax Output: 10
```

Training is done in batches of 200 using TensorFlow's implementation of the [Adam Optimizer](https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer) to minimize cross entropy.

*during training, pkeep for the fully connected + dropout layer is set to 0.75*

### Training Data

In order to generate training and testing data, you can use the `generate.py` CLI which will perform segmentation on an input image and allow you to discard symbols that were segmented incorrectly.

For example, the following image (from `source/`) was used to generate training data for the `dollar` symbol:

![dollar_train](source/train_dollar.jpeg)

The script will then crop each symbol to  and generate an image at each 15 degree rotation like so:

|![](examples/rot/1.jpg)|![](examples/rot/2.jpg)|![](examples/rot/3.jpg)|![](examples/rot/4.jpg)|![](examples/rot/5.jpg)|![](examples/rot/6.jpg)|
|---|---|---|---|---|---|
|![](examples/rot/7.jpg)|![](examples/rot/8.jpg)|![](examples/rot/9.jpg)|![](examples/rot/10.jpg)|![](examples/rot/11.jpg)|![](examples/rot/12.jpg)|
|![](examples/rot/13.jpg)|![](examples/rot/14.jpg)|![](examples/rot/15.jpg)|![](examples/rot/16.jpg)|![](examples/rot/17.jpg)|![](examples/rot/18.jpg)|
|![](examples/rot/19.jpg)|![](examples/rot/20.jpg)|![](examples/rot/21.jpg)|![](examples/rot/22.jpg)|![](examples/rot/23.jpg)|![](examples/rot/24.jpg)|
