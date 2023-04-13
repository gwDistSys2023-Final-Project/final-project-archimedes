# Distributed Machine Learning Inference for Real Time Trajectory Predictions

## Introduction
Machine learning has been advancing rapidly over the years, and one of its most promising applications is predicting the future movements of objects such as cars, planes, and drones based on their past trajectories. This type of prediction is crucial for various applications, including autonomous driving, air traffic control, and drone delivery services. In this project, we aim to build a distributed machine learning inference system for real-time trajectory predictions. This report will discuss the importance of this project, what we specifically aim to build or design, the main challenges we will face, and some sample papers that will guide our work.

## Importance of the Project
Predicting future trajectories of objects is critical for the safety and efficiency of various applications. Autonomous vehicles require accurate trajectory predictions to make informed decisions about their movements, and air traffic control needs to predict the movements of planes to ensure safety and reduce congestion. In addition, drone delivery services rely on trajectory predictions to ensure timely and efficient deliveries. Therefore, building a distributed machine learning inference system for real-time trajectory predictions is crucial for the success of these applications. Moreover, such application is associated with a large input of real-time geo-encoded data. This creates the need to maximize the throughput of the model for it to be able to predict more locations per unit time.

## Design and Build
To achieve the goal of building a distributed machine learning inference system for real-time trajectory predictions, we will use a GRU machine learning model and will use distributed systems to leverage the performance of the already created model. The already trained model learns the trajectories pattern of the city of Daejeon from taxi data inputted. It predicts the next trajectory based on the previously visited locations. The layers of this sequential model are partitioned and distributed across different nodes. Each node will compute part of the inference and then communicate the result to the next node for it to do its task. The distributed system will provide the same accuracy while having an increased throughput and delivering the results with almost no latency.

## Challenges
The following are the main distributed systems challenges we will face in building a distributed machine learning inference system for real-time trajectory predictions.
- **Throughput**: The distributed ML model needs to handle more requests per unit time than the non-distributed one. This is one of the advantages of distributing the inference. One of the challenges of this project is succeeding in increasing throughput.
- **Latency**: The system must provide predictions in almost real-time response. This challenge will be addressed by using a distributed system that can provide predictions within the required timeframe by optimizing the data processing and communication pipeline.
- **Load balancing and system configuration**: In order to overcome the previously mentioned challenges. An optimal system architecture should be used. This is a challenge by itself due to the different possible combinations in the partitioning and configuration of the system. A correct load balance on the different nodes and an adequate configuration are necessary for an optimal performance.
- **Network overhead and payload**: Optimizing the network overhead and payload is key to obtaining an optimal result. This is expected to be a challenge in our application.

## References
- [1] A. Parthasarathy and B. Krishnamachari, "DEFER: Distributed Edge Inference for Deep Neural Networks," 2022 14th International Conference on COMmunication Systems & NETworkS (COMSNETS), Bangalore, India, 2022, pp. 749-753, doi:
10.1109/COMSNETS53615.2022.9668515.
- [2] Zhuoran Zhao, Andreas Gerstlauer. (2018) DeepThings: Distributed Adaptive Deep Learning
Inference on Resource-Constrained IoT Edge Clusters.
http://slam.ece.utexas.edu/pubs/codes18.DeepThings.pdf
- [3] Ivan Rodriguez-Conde, Celso Campos, Florentino Fdez-Riverola. (2023) Horizontally Distributed Inference of Deep Neural Networks for AI-Enabled IoT.
https://www.mdpi.com/1424-8220/23/4/1911
- [4] Lunga, D., Gerrand, J., Yang, L., Layton, C., & Stewart, R. (2020). Apache Spark accelerated deep learning inference for large scale satellite image analytics. IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 13, 271–283.
https://doi.org/10.1109/jstars.2019.2959707
- [5] C. Hu and B. Li, "Distributed Inference with Deep Learning Models across Heterogeneous Edge Devices," IEEE INFOCOM 2022 - IEEE Conference on Computer Communications, London, United Kingdom, 2022, pp. 330-339, doi: 10.1109/INFOCOM48880.2022.9796896.
https://iqua.ece.toronto.edu/papers/chenghao-infocom22.pdf
- [6] S. Henna and A. Davy, "Distributed and Collaborative High-Speed Inference Deep Learning or Mobile Edge with Topological Dependencies" in IEEE Transactions on Cloud Computing, vol.10, no. 02, pp. 821-834, 2022.
https://doi.ieeecomputersociety.org/10.1109/TCC.2020.2978846
- [7] Chinchali, S. P., Cidon, E., Pergament, E., Chu, T., & Katti, S. (2018). Neural networks meet physical networks. Proceedings of the 17th ACM Workshop on Hot Topics in Networks.
https://doi.org/10.1145/3286062.3286070
- [8] N. Li, A. Iosifidis and Q. Zhang, "Distributed Deep Learning Inference Acceleration using Seamless Collaboration in Edge Computing," ICC 2022 - IEEE International Conference on Communications, Seoul, Korea, Republic of, 2022, pp. 3667-3672, doi:
10.1109/ICC45855.2022.9839083.
- [9] F. M. C. de Oliveira and E. Borin, “Partitioning Convolutional Neural Networks to Maximize the Inference Rate on Constrained IoT Devices,” Future Internet, vol. 11, no. 10, p. 209, Sep. 2019, doi: 10.3390/fi11100209.
