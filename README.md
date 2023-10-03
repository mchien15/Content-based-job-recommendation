# PROJECT CHO HỌC PHẦN IT3190 - DATA MINING - HUST
## CONTENT BASED JOB RECOMMENDATION SYSTEM  
### Hướng dẫn cài đặt 
+ Sử dụng [anaconda](https://www.anaconda.com/) để tạo virtual environment.
+ Bật terminal chạy command sau: `conda activate`.
+ Tạo virtual environment có tên `datamining`: `conda create --name datamining python=3.8`.
+ Chạy command: `conda activate datamining`, để khởi động virtual environment `datamining` vừa tạo.
+ Clone src code về máy `git clone https://github.com/duongngyn0510/Content-based-job-recommendation.git` (lưu ý phải cài đặt [Gitbash](https://git-scm.com/downloads)).
+ Change dir để tới folder sau khi clone: `cd Content-based-job-recommendation`.
+ Do tập data lớn hơn so với giới hạn quả github nên sẽ tải [data](https://www.kaggle.com/datasets/kandij/job-recommendation-datasets) trực tiếp về máy. Sẽ có 4 file data cần tải đó là: `Combined_Jobs_Final.csv`, `Experience.csv`, `Job_Views.csv` và `Positions_Of_Interest.csv`.
+ Tạo thêm folder `data` tại chính folder `Content-based-job-recommendation` rồi copy 4 file vừa tải xuống vào folder `data`.
+ Cài đặt thư viện: bật terminal chạy command: `pip install -r requirements.txt`.
+ Chạy file `recommend.ipynb` để thực hiện đề xuất cho các user.

**Lưu ý**: Để có thể chạy để xuất thành công cho các Applicant.ID nhiều lần thì máy phải có tối thiếu RAM 8GB

### Giải thích chi tiết các file
+ `bert_pretrained_vector`: chứa feature vector pretrained **BERT** của JOB POSITION, trong đó `pos_feature_vector.pt` sử dụng pretrained BERT để feature extraction.
+ `data`: chứa data ban đầu, trong đó `Combined_Jobs_Final.csv` có mục đích để tạo job corpus, `Experience.csv`, `Job_Views.csv` và `Positions_Of_Interest.csv` để tạo user corpus
+ `clean_data` chứa data sau khi được clean, trong đó `final_df_jobs.csv` là data của job corpus, `final_user_df.csv` là data của user corpus
+ `npz_file`: chứa feature vector của tập job, trong đó `count_jobid.npz`: sử dụng **Bag-of-word**, `tfidf_jobid.npz` sử dụng **Tf-idf** để feature extraction.
+ `pkl_file`: chứa các file pkl.
+ `bert_extract_feature.py`: file feature extraction sử dụng BERT.
+ `clean.py`: file clean data.
+ `extract_feature.py`: file feature extraction cho JOB và USER (sử dụng TF-IDF, BOW, KNN, BERT).
+ `job_corpus.iypnb`: file tạo job corpus từ `Combined_Jobs_Final.csv`.
+ `user_corpus.iypnb`: file tạo user corpus từ `Experience.csv`, `Job_Views.csv` và `Positions_Of_Interest.csv`.
+ `requirements.txt`: chứa các thư viện cần thiết.
+ `utils.py`: chứa các function cần thiết.
+ `recommend.ipynb`: file đề xuất các job cho từng `Applicant.ID`


Đề có thể đề xuất cho các `Applicant.ID`, ta chạy từng cell trong `recommend.ipynb`.

### Kết quả
Với người dùng có  `Applicant.ID = 326` với vị trí kinh nghiệm là `java developer`.
+ Recommendation sử dụng tf-idf
!['Recommendation sử dụng tf-idf'](https://github.com/duongngyn0510/Content-based-job-recommendation/blob/master/imgs/tfidf-rec.png)

+ Recommendation sử dụng bow
!['Recommendation sử dụng bow'](https://github.com/duongngyn0510/Content-based-job-recommendation/blob/master/imgs/bow-rec.png)

+ Recommendation sử dụng KNN
!['Recommendation sử dụng KNN'](https://github.com/duongngyn0510/Content-based-job-recommendation/blob/master/imgs/knn-rec.png)

+ Recommendation sử dụng pretrained BERT
!['Recommendation sử dụng pretrained BERT'](https://github.com/duongngyn0510/Content-based-job-recommendation/blob/master/imgs/bert-rec.png)
