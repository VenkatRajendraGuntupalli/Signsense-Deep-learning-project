# SignSense: Deep Learning-Driven Caution Detection for Unmarked Roads

**Project Final Report for CPS 584 - Advanced Intelligent Systems and Deep Learning**

**Prof: Mehdi R. Zargham**

**Master’s in Computer Science**

**Submitted By:**
- Venkat Rajendra Guntupalli (101744701)

## Table of Contents
- [Introduction](#introduction)
- [Project Description](#project-description)
- [Summary of Tasks](#summary-of-tasks)
  - [Data Collection and Preprocessing](#data-collection-and-preprocessing)
  - [Model Architecture Selection](#model-architecture-selection)
  - [Model Training](#model-training)
  - [Feature Extraction](#feature-extraction)
  - [Evaluation and Testing](#evaluation-and-testing)
- [Source of Data Collection](#source-of-data-collection)
- [Proposed Method](#proposed-method)
- [Details of Implementation](#details-of-implementation)
- [Results](#results)
- [Conclusion](#conclusion)

## Introduction

Road safety is a paramount concern worldwide, and the absence of clear indications and signage on unmarked roads poses significant challenges in ensuring safe transportation. In response to this pressing issue, we present "SignSense," an innovative deep learning-driven approach designed to identify and recommend the appropriate road caution signs for unmarked roads. By leveraging cutting-edge deep learning techniques, SignSense aims to mitigate the risks associated with unmarked roads and enhance overall road safety.

## Project Description

By leveraging a diverse dataset encompassing various road conditions, the deep learning model is trained to recognize potential hazards. Through fine-tuning, the model identifies the specific caution sign corresponding to each detected hazard. The project's objective is to provide actionable recommendations to road safety authorities, enabling them to proactively address the safety concerns of unmarked roads by promptly installing the appropriate caution signs.

## Summary of Tasks

### Data Collection and Preprocessing

One of the significant challenges was curating a diverse and comprehensive dataset of road images, including curve roads, straight roads, junction-type roads, and uneven roads. This required meticulous attention to detail to ensure the dataset's accuracy and reliability.

### Model Architecture Selection

We built a custom Convolutional Neural Network (CNN) from scratch to tackle the multi-image classification task. Through careful experimentation and diligent hyperparameter tuning, we developed a robust CNN architecture that demonstrated strong capabilities in hazard detection and caution sign recognition.

### Model Training

We encountered a class imbalance in the dataset, leading to biased model performance. To address this, we employed data augmentation techniques to balance the class distribution. This significantly improved the model's convergence rate and generalization capabilities.

### Feature Extraction

We used CNNs for feature extraction, gradually increasing the network's depth and tuning the number of filters. This improved the model's ability to recognize intricate visual patterns associated with road hazards.

### Evaluation and Testing

We meticulously designed a comprehensive test set, covering a wide range of road conditions and caution signs. The model demonstrated impressive accuracy and precision in detecting potential hazards and recommending appropriate caution signs.

## Source of Data Collection

Initially, we relied on public datasets and online repositories. To enhance the dataset, we applied data augmentation techniques and leveraged crowdsourcing platforms to gather more images from various geographical regions. This ensured a diverse and balanced dataset representing real-world road conditions.

## Proposed Method

We chose CNNs for their effectiveness in image recognition tasks and their ability to automatically learn hierarchical representations of features from data. This approach reduces the need for manual feature engineering and effectively handles spatial dependencies in images.

## Details of Implementation

We used TensorFlow as our primary deep learning framework and Google Colab as our Integrated Development Environment (IDE). Google Colab's GPU resources significantly reduced the model's training time. We relied on several libraries, including TensorFlow, NumPy, Matplotlib, OpenCV, and scikit-learn, to support our deep learning implementation.

## Results

Our model successfully detects road hazards and recommends appropriate caution signs. For example, an input with an S curve road results in an S curve caution sign, and a straight road gets a speed limit caution sign to prevent accidents due to over-speeding.

![Example Result](path_to_example_image)

## Conclusion

Our deep learning-driven SignSense project has successfully addressed the challenges of road safety on unmarked roads. Through meticulous data collection, preprocessing, and iterative model architecture design, we have demonstrated the model's efficacy in hazard detection and caution sign recommendation. Moving forward, we plan to collaborate with road safety authorities for real-world validation and explore advanced CNN architectures and transfer learning strategies.

---
