# BITCOIN PRICE PREDICTION USING PYSPARK

Bitcoin Price Prediction using Spark Global and self-designed Local Model with Big data preprocessing and manipulation solution.

* Global Model: Spark build-in MLlib, model can benefit from all the data.
* Local Model: Utilize the ML algorithm from third party(eg. scikit-learn), model only can benefit from a subset of the data, but could be faster.


## Prerequisites

- Packages:  
  * python >= 3.8.8
  * pyspark >= 3.1.1
  * numpy >= 1.19.2
  * pandas >= 1.2.3
  * plotly >= 4.14.3
  * scikit-learn >= 0.24.1
  * statsmodels >= 0.12.2
  * pmdarima >= 1.8.2


## Built With

* [Spark](https://spark.apache.org/) - Lightning-fast unified analytics engine
* [Python](https://www.python.org/) - Programming language


## Files
### code
- global_mode.ipynb :  Global Model Prediction on Spark.
- local_LR.ipynb :  Local Model design(Linear Regression) on Spark to make predictions.
- local_autoReg.ipynb :  Local Model design(ARIMA/VectorARIMA) on Spark to make predictions.
- preprocess_bitcoin_pyspark.ipynb :  Data imputation and resampling by big data solution(Spark).
- blockChain_crawler.ipynb :  A crawler to get BlockChain Information.
- feature_engineering.ipynb :  Feature Engineering, include Data combination, Label maker, financial indicators maker.
- tsCrossValidation.py :  A common code for time series Cross Validation.
### [dataset](https://www.kaggle.com/mczielinski/bitcoin-historical-data)
- bitcoin_1m_1min.csv : A subset dataset for functionality test; 1 month(03/2021) 1min interval data of bitcoin.
### Paper
- conference_paper.pdf

## Authors

* **Chi Wang**
* **Luer Lyu**
* **Joel Ligma**
* **Junfeng Wang**


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
If you want to cooperate or use this project, please contact the author: wangchi.work@gmail.com


## Acknowledgments

[1] S. Nakamoto, “Bitcoin: A Peer-to-Peer Electronic Cash System,” p. 9. 
[2] E. Akyildirim, A. Goncu, and A. Sensoy, “Prediction of cryptocurrency returns using machine learning,” Ann. Oper. Res., vol. 297, no. 1, pp.
3–36, Feb. 2021, doi: 10.1007/s10479-020-03575-y.
[3] J. H. F. Flores, P. M. Engel, and R. C. Pinto, “Autocorrelation and partial autocorrelation functions to improve neural networks models on univariate time series forecasting,” in The 2012 International Joint Conference on Neural Networks (IJCNN), Jun. 2012, pp. 1–8. doi: 10.1109/IJCNN.2012.6252470.
[4] C.-H. Wu, C.-C. Lu, Y.-F. Ma, and R.-S. Lu, “A New Forecasting Framework for Bitcoin Price with LSTM,” in 2018 IEEE International Conference on Data Mining Workshops (ICDMW), Nov. 2018, pp. 168–175. doi: 10.1109/ICDMW.2018.00032.
[5] I. Yenidog ̆an, A. C ̧ayir, O. Kozan, T. Dag ̆, and C ̧. Arslan, “Bitcoin Forecasting Using ARIMA and PROPHET,” in 2018 3rd International Conference on Computer Science and Engineering (UBMK), Sep. 2018, pp. 621–624. doi: 10.1109/UBMK.2018.8566476.
[6] W. Waheeb, H. Shah, M. Jabreel, and D. Puig, “Bitcoin Price Forecast- ing: A Comparative Study Between Statistical and Machine Learning Methods,” in 2020 2nd International Conference on Computer and Information Sciences (ICCIS), Oct. 2020, pp. 1–5. doi: 10.1109/IC- CIS49240.2020.9257664.
[7] S. McNally, J. Roche, and S. Caton, “Predicting the Price of Bitcoin Using Machine Learning,” in 2018 26th Euromicro International Con- ference on Parallel, Distributed and Network-based Processing (PDP), Mar. 2018, pp. 339–343. doi: 10.1109/PDP2018.2018.00060.
[8] H.Kavitha,U.K.Sinha,andS.S.Jain,“PerformanceEvaluationofMa- chine Learning Algorithms for Bitcoin Price Prediction,” in 2020 Fourth International Conference on Inventive Systems and Control (ICISC), Jan. 2020, pp. 110–114. doi: 10.1109/ICISC47916.2020.9171147.
[9] A. Adebiyi, A. Adewumi, and C. Ayo, “Stock price prediction using the ARIMA model,” presented at the Proceedings - UKSim-AMSS 16th International Conference on Computer Modelling and Simulation, UKSim 2014, Mar. 2014. doi: 10.1109/UKSim.2014.67.
[10] D. U. Sutiksno, A. S. Ahmar, N. Kurniasih, E. Susanto, and A. Leiwakabessy, “Forecasting Historical Data of Bitcoin using ARIMA and ↵-Sutte Indicator,” J. Phys., p. 5.
[11] “NEURAL NETWORK MODEL VS. SARIMA MODEL IN FORE- CASTING KOREAN STOCK PRICE INDEX (KOSPI),” Issues Inf. Syst., 2007, doi: 10.48009/2 iis 2007 372-378.
[12] N. Merh, V. Saxena, and K. Pardasani, “A comparison between Hybrid Approaches of ANN and ARIMA for Indian Stock Trend Forecasting,” Bus. Intell. J., vol. 3, pp. 23–44, Jul. 2010.
[13] S. C. Purbarani and W. Jatmiko, “Performance Comparison of Bitcoin Prediction in Big Data Environment,” in 2018 International Workshop on Big Data and Information Security (IWBIS), May 2018, pp. 99–106. doi: 10.1109/IWBIS.2018.8471691.
[14] Y. Go ̈ru ̈r, BITCOIN PRICE DETECTION WITH PYSPARK USING RANDOM FOREST. 2018. doi: 10.13140/RG.2.2.26508.16008.
[15] P.Ciaian,M.Rajcaniova,andd’ArtisKancs,“TheeconomicsofBitCoin price formation,” Appl. Econ., vol. 48, no. 19, pp. 1799–1815, Apr. 2016, doi: 10.1080/00036846.2015.1109038.
[16] Z. Chen, C. Li, and W. Sun, “Bitcoin price prediction using ma- chine learning: An approach to sample dimension engineering,” J. Comput. Appl. Math., vol. 365, p. 112395, Feb. 2020, doi: 10.1016/j.cam.2019.112395.
[17] W. Chen, H. Xu, L. Jia, and Y. Gao, “Machine learning model for Bitcoin exchange rate prediction using economic and technology deter- minants,” Int. J. Forecast., vol. 37, no. 1, pp. 28–43, Jan. 2021, doi: 10.1016/j.ijforecast.2020.02.008.
[18] H. Jang and J. Lee, “An Empirical Study on Modeling and Pre- diction of Bitcoin Prices With Bayesian Neural Networks Based on Blockchain Information,” IEEE Access, vol. 6, pp. 5427–5437, 2018, doi: 10.1109/ACCESS.2017.2779181.
[19] M. Saad and A. Mohaisen, “Towards characterizing blockchain-based cryptocurrencies for highly-accurate predictions,” in IEEE INFOCOM 2018 - IEEE Conference on Computer Communications Workshops (INFOCOM WKSHPS), Apr. 2018, pp. 704–709. doi: 10.1109/INF- COMW.2018.8406859.
[20] T. Zeng, M. Yang, and Y. Shen, “Fancy Bitcoin and conventional financial assets: Measuring market integration based on connected- ness networks,” Econ. Model., vol. 90, pp. 209–220, Aug. 2020, doi: 10.1016/j.econmod.2020.05.003.
[21] G. Giudici, A. Milne, and D. Vinogradov, “Cryptocurrencies: market analysis and perspectives,” J. Ind. Bus. Econ., vol. 47, no. 1, pp. 1–18, Mar. 2020, doi: 10.1007/s40812-019-00138-6.
[22] M.Matta,I.Lunesu,andM.Marchesi,“BitcoinSpreadPredictionUsing Social And Web Search Media,” p. 10.
[23] C.BergmeirandJ.M.Ben ́ıtez,“Ontheuseofcross-validationfortime series predictor evaluation,” Inf. Sci., vol. 191, pp. 192–213, May 2012, doi: 10.1016/j.ins.2011.12.028.