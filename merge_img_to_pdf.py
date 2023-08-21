import os

import img2pdf

import os
import time


def sort_files(directory):
    """Sorts the files in the specified directory by creation date."""
    files = os.listdir(directory)
    files = [os.path.join(directory, file) for file in files]
    # files.sort(key=lambda file: os.path.getctime(file))
    files.sort()
    return files


def merge_images_to_pdf(image_paths):
    pdf = img2pdf.convert(image_paths)
    with open("prompt_engineering_coursera.pdf", "wb") as f:
        f.write(pdf)


if __name__ == "__main__":
    # # list all files present in the "F:\Locker\EN\Topics\GPT\PromptEngineering\PromptEngineeringForChatGPT_Coursera" directory
    # files = os.listdir("F:\Locker\EN\Topics\GPT\PromptEngineering\PromptEngineeringForChatGPT_Coursera")
    #
    # # create a list of file paths for all files in the "F:\Locker\EN\Topics\GPT\PromptEngineering\PromptEngineeringForChatGPT_Coursera" directory
    # file_paths = []
    # for file in files:
    #     file_paths.append(
    #         os.path.join("F:\Locker\EN\Topics\GPT\PromptEngineering\PromptEngineeringForChatGPT_Coursera", file))
    # print(file_paths)
    #
    # # sort the file by creation date
    # file_paths.sort(key=os.path.getctime)

    # image_paths = ["image1.jpg", "image2.jpg", "image3.jpg"]
    merge_images_to_pdf(sort_files("F:\Locker\EN\Topics\GPT\PromptEngineering\PromptEngineeringForChatGPT_Coursera"))
