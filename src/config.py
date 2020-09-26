import datetime

#============================ Paths ==========================

DATA_DIR = 'data/0_raw/COVID-19 Radiography Database'
PROCESSED_DATA_PATH = 'data/3_processed/data.csv'
PRETRAINED_MODEL = 'output/models/inference/base_model_covid.h5'

#============================ Live Data config ==========================

# https://rapidapi.com/astsiatsko/api/coronavirus-monitor
rapid_api_key = "dd8d4e05e8mshc5ab62dcd8a5f08p14b028jsna2726a63a74d"


#============================ Samples Links ==========================
sample_images_dict = {
    'covid_sample1':'https://storage.googleapis.com/kagglesdsdata/datasets%2F576013%2F1042828%2FCOVID-19%20Radiography%20Database%2FCOVID-19%2FCOVID-19%20%28100%29.png?GoogleAccessId=databundle-worker-v2@kaggle-161607.iam.gserviceaccount.com&Expires=1601209289&Signature=ZALqHPsMcKZyN2I1xggUvU9DU5yHSDWDol3Kje6O0fqjw8jrFITv5LAzrjDiz%2Byi1h6y2gO4cZSWbXqhf3PmhyxNJXasyLFHotSSokJD0BPoCvfSDe7eaXMnMeDArqpKxZ4pBjWqw97KZs20%2FVgnp5mdVaaurlv%2BdN%2FXHg9HtC6nw7G1I3EEHcVmAM0q0%2BZHE02w%2Bo%2BM903W5584gFchLo3hTTGdLSGn%2FWUNRJMPTewg%2F9os0w76O7x2l0PFWz%2Fb4RylrPUYp2wa0Tzz6l3hrFQjYXoKneqfySZ51h8pFMSiIi%2Fjs%2BCbLXNeRy39BMlgDA%2FcFEHKEiP%2FyEMRJYBdZQ%3D%3D',
    'covid_sample2':'https://storage.googleapis.com/kagglesdsdata/datasets%2F576013%2F1042828%2FCOVID-19%20Radiography%20Database%2FCOVID-19%2FCOVID-19%20%28101%29.png?GoogleAccessId=databundle-worker-v2@kaggle-161607.iam.gserviceaccount.com&Expires=1601209294&Signature=pe9euyowyJddlfZrrEtHgj0yFLjxbZyf7wsZcFz%2B%2F82KDTtbM7sDQ81bCb4bPn88Xu1LKd71fqannJqA3ePP%2FyMU2llXANXnjpK0ConaLD6scgTDQiaoa04MqVxwjrkGrrZd%2FJQhyUBG13l4ibBFiRjXNQ1BnWUPnEJMWq%2Bcg%2FJcBEX2OLXW1uFKhxOBDFDUeo%2FV71eJ67ISqxhLYU8hB0bMZB%2Bc202Aeu6dXS2xTAMGevebYI3ySV3SgBRBu060K7toVxtwxUmR6WshBtSButMJPEC6VcTw2HbLGShKPdcghvsZGdqBdisLwO6355mxHAmqZnansfalQ6xu%2F0%2F6zg%3D%3D',
    'covid_sample3':'https://storage.googleapis.com/kagglesdsdata/datasets%2F576013%2F1042828%2FCOVID-19%20Radiography%20Database%2FCOVID-19%2FCOVID-19%20%28102%29.png?GoogleAccessId=databundle-worker-v2@kaggle-161607.iam.gserviceaccount.com&Expires=1601209299&Signature=UJx%2B8isPTvVAWA6bt2HyiFfJ9L%2B3Ft0JgC1o9Yk9WFwFu3dnEoxo9IUP5P7yyzXi8yiUv00tMF60mv3EQVhpXxL4Ud8vLChBNTOcf41lA4MVldAb9ADoNvDqCSoaxQg9wkK4F7Sdst%2B7SphN2BWnfRCQz8BJB5oxGvTf6GMW%2BF%2BDIf876c1oboqnKVo7V9Uz8mjPGBvg3ZkCUay26YmOEpshe64pVwUe4o5TlcItfEChWN5wRQS%2BHZgb00kSlwqFCP%2F55YFLtap%2F9M0XmMHMsp%2BczNKRMQU6%2BV6GkW5zTf491hsT1z%2FXKGxQr3TLAOrJdsY5rILdeQNaergl3G0pMw%3D%3D',
    'covid_sample4':'https://storage.googleapis.com/kagglesdsdata/datasets%2F576013%2F1042828%2FCOVID-19%20Radiography%20Database%2FCOVID-19%2FCOVID-19%20%2810%29.png?GoogleAccessId=databundle-worker-v2@kaggle-161607.iam.gserviceaccount.com&Expires=1601209279&Signature=OVdG5mTlpZjgDMj%2BnrCX2e0IPplsiUkA5ds%2BGfgSOBT5yndPDdwZULG%2Fzk5VMUMughLbmKX3u0rkW%2BR3cnUlBAEk5nOsppVNdfN06HzV1t0Q7qGbMRWY9y6KUUHt4YbG5%2BYdV6AkjLX3Odeg5Y%2BRe0r2my%2FavEUwqnCE2cLNfTLT%2FeJcSbh9q76Q3UFreV21NJcG93vAtJXWu5xaiFW1dyxL2K0ZyKFhQbtLoLxStVxTKOWtk3G41jqdPJH9q04ZWESjyT%2Fe43NXQxNrasB1PEVjFvoBXVsifC7nCardqXJG7AwWRKY6VyGJg2XVTb6BoT5y9vbIenkPydH%2BNz1hag%3D%3D',
    'covid_sample5':'https://storage.googleapis.com/kagglesdsdata/datasets%2F576013%2F1042828%2FCOVID-19%20Radiography%20Database%2FCOVID-19%2FCOVID-19%20%28104%29.png?GoogleAccessId=databundle-worker-v2@kaggle-161607.iam.gserviceaccount.com&Expires=1601211335&Signature=KmC6AyZB71aZRlr73kPE9gTHtwSvXtAXZQLuTqmmxgHC6rk57U1hO65xMP0NOu53pOARX1lrI5ic7AetMxXWP9I1XJnC8PWW1t9IfVOegNfOfv4GlaGhBoymlvfrMybLHQHad7z7Aitz%2BJ1n1LrdfwDiMLkru5DoNtFZ%2FTqwSgir7qBe13KvyIe%2Fgr%2B%2BHcfaTK%2BoVmToQna5QBdzI2P3hVJc%2B%2FpFlsohjMyfp8Ops14hgbidTX2AEKh6S%2FXZGgJmwXiwumPQfcKhYaa7ZJqghOS9XMnimIor52J8kEV%2FvuJQ3g8rQUMFBUrol%2FL1OgRbL4F%2BitGuUK9ZEl0MUKDGyw%3D%3D',
    'covid_sample6':'https://storage.googleapis.com/kagglesdsdata/datasets%2F576013%2F1042828%2FCOVID-19%20Radiography%20Database%2FCOVID-19%2FCOVID-19%20%28106%29.png?GoogleAccessId=databundle-worker-v2@kaggle-161607.iam.gserviceaccount.com&Expires=1601211345&Signature=p6v6e9C1eBfgHQSheWp1fHwvoLQyUG%2FWEYzQH0EPJWFlgYOIt5qhCWBaVozAySx%2BJgicddbxP8u5onwhxRj61H7fLwwAI1LHqQvphzjzbUMYAipyDaCOZKGF3me9%2FeTU5XXP3mfSZr3Eur0JU4HChRGFH0%2Bs%2FqS4LuS5iXEuk3kqpXL7ReUHxqlA421tLK%2Bk4LvyQntKt2CvO1%2BK6tMpKG1mzFnOT8wxhg2LBDgU82x%2FpKrhCkPLUBD7XqG2pmOs6Kdqe1GHnRB%2Bu1ZqU0GhteRpFjU6DzD7%2FszxtoGFw2%2FsThVHfo1GFcKeRnZ%2BG4yvVAwz0JCo4w69XfI%2Bq8o%2F%2Fg%3D%3D',
    'normal_sample1':'https://storage.googleapis.com/kagglesdsdata/datasets%2F576013%2F1042828%2FCOVID-19%20Radiography%20Database%2FNORMAL%2FNORMAL%20%281%29.png?GoogleAccessId=databundle-worker-v2@kaggle-161607.iam.gserviceaccount.com&Expires=1601213984&Signature=DUdzJUGvruzE70vY4ZHO7xKUodsbv8Qz2IhzRIMFNftQdqNP2%2BpTvgWbOpmZ40zGhoYnAMBYQFWkA917dxDN9uPWsJW4wK4S%2BJC780XnZ9HDHsbKQKaoyysiFAFMfft2DXuU3zdbZyNfacP2VaXa1SqsT%2FPzpbUT5xna0%2B7bOvx%2Bw4fxs9U0vxVu6g3n%2F6oElzUFAqx7Lzb6hC0ZKs83fpwpFILE7oftOiTL5%2BJe%2ByG3qIUs%2FSHKUptTFnFvqSUd67EAmPgNIRmt6zJaY6LInY%2Bl0ykRTeLBEVv8La0iLqoW3yXOmwowZ7WNNwvY3TuN%2FgpIaoMCcjkh0MJ2Xpl9YQ%3D%3D',
    'normal_sample2':'https://storage.googleapis.com/kagglesdsdata/datasets%2F576013%2F1042828%2FCOVID-19%20Radiography%20Database%2FNORMAL%2FNORMAL%20%2810%29.png?GoogleAccessId=databundle-worker-v2@kaggle-161607.iam.gserviceaccount.com&Expires=1601213984&Signature=aedGU942lThwf%2BAVS7IDP4BlWOmZQ6TFL8f60pB5nutpIJl02kNRYpL45UAB8aVaBnOL6wbjJMWosiE9eLxTJQYABAiruUCSH1ZRq8w99nLBmj5ppQ9TUf8w2lRFF96BHsd%2BK3phmmXHp50ef1pC4COQqkL3dcFHln%2FGFPBfz4D%2FSwTrpZJeZzXMD2vnkDW8LuxoPXVdn1ZvCr6ndUYQvYuQ9V0R7apRXkLTwxxKN6bCaOPj7E%2FVvKQuqFgCoJwriBB7mVqaln%2FyCRutLbolTKAwRLpYwxhbZXD3WT0Df2JE96e1AYgF7SsOpfIlm3DqVP7BSFmFR9xvoauCYjZsQw%3D%3D',
    'normal_sample3':'https://storage.googleapis.com/kagglesdsdata/datasets%2F576013%2F1042828%2FCOVID-19%20Radiography%20Database%2FNORMAL%2FNORMAL%20%28100%29.png?GoogleAccessId=databundle-worker-v2@kaggle-161607.iam.gserviceaccount.com&Expires=1601213984&Signature=qN5P5PJDGI%2BaD08jdq%2F9FOCPs4lXHJKc615qAeXK%2F0Lpb7IbZ%2F%2F2QDqDtjyAb9Nz30u6eX2%2B6X%2FAXXZ7f5aiq16tZNiadFQnSptrFq9tpm3oAtB5wcqbYxwD6ILXGV13chliqyJeAKGo4OgZuoAhJj8cRQ65wTo35QkyMsSgzAwrj8FMJdPxoHpXiTc9WVa6QawlKFhj9gqc4PFDSWqGiEEE3u9k1PI2Sh04Vquhh%2FKHWlnYraRJIFG5CjIi0rFa34VSvF8k9afb%2BkZD5DJLjwXr%2BBTuRcQXwElFYwJjvpJirqrV05hq5dJ0iXyW4V1w5YSgsQNHs1CUrJIFFktgFQ%3D%3D',
    'normal_sample4':'https://storage.googleapis.com/kagglesdsdata/datasets%2F576013%2F1042828%2FCOVID-19%20Radiography%20Database%2FNORMAL%2FNORMAL%20%281000%29.png?GoogleAccessId=databundle-worker-v2@kaggle-161607.iam.gserviceaccount.com&Expires=1601213984&Signature=aQIeptFf0ZRA8bdtZfuahWuRZ2VJfS55OpL%2Fzh%2BLl%2FnqdoU8kMVaw7ROnSXdRT7AyDBIIeLGmbglwROF64WoKat3ePC6Lnl5HgtfholFbKm96CNUDdNYYwpVi0dLP4e75hWR%2F7hLvARdSK7t6sZM5DBNlD8ngrsokNimgfOYVDJhfuIm9NJufMljGw40eGLeIAeZJnoXDDVbN6K5iLxABZ0q00A2lipYAjDEZqshW7kbGlvr0z0884TZ2kkqpGj4ardFQgRV0KzOEUK9q3We9VfoHuDj6D%2FG%2BJ5dK3Aurmo6%2BuAO%2FYV4KKE%2B2MPi1frOHhFe2kfmxh8dxbQmc59JDg%3D%3D',
    'normal_sample5':'https://storage.googleapis.com/kagglesdsdata/datasets%2F576013%2F1042828%2FCOVID-19%20Radiography%20Database%2FNORMAL%2FNORMAL%20%281001%29.png?GoogleAccessId=databundle-worker-v2@kaggle-161607.iam.gserviceaccount.com&Expires=1601213984&Signature=nrqJMBrNRrO6G2LhrShisvmbT3CvC3Qz3CWaGm7pRYVBKi%2FyakVUBps5G416%2FJ%2BWw8494LrVCBWatp0LQnLr4OSG3EuoRy%2Bbfd77xBKbjB1KMYEEGsRhF%2F2qs3JpOcqEwTeb%2FIoSSTDE8naEvF1Kzkka0ZW810wByKCgNsWbGzGGUuOtBvIOKLFS%2Fr2seJ90SLvyFNqH8y0niyuMJRoGcbpTMcoidc7Lj4IOooqE9Tsz%2B2A47HKnt73x%2FDVyzVKi0SAKkouVULeX6BGZYtZyu2e2legiijnVryHPRGfoAtNlsAUeYkYdMaAqoovnJHrZ6f%2BdkyGi6PBrtghQF0GzOg%3D%3D',
    'pneumonia_sample1':'https://storage.googleapis.com/kagglesdsdata/datasets%2F576013%2F1042828%2FCOVID-19%20Radiography%20Database%2FViral%20Pneumonia%2FViral%20Pneumonia%20%281%29.png?GoogleAccessId=databundle-worker-v2@kaggle-161607.iam.gserviceaccount.com&Expires=1601214014&Signature=F68kLiUvRI%2B5BTLdnbcRvLtWM5spuhVCoDiSf9tvQugMQ3t%2FXK5%2BbxSmIDfRZ4g2vsQ4vEYw2F0ugzsmzNdXGMrULU4qGXbAxhqGM4EQPPSqKMgQIKKY%2BLsLIPhmTF1IBwlGbCDGr8C4uXRdcPH49HUo3OBhoNPSz8Rutw1iiwZKdA%2BWaSBJlJFWsFZ952wrkJia1oVeH5CQWVcR2H7%2FxpFVaW%2FvmGChVbkJQGieb9oOmSlb0q4iIp%2FU0TGTLC3a565gKK1NPr%2BRIOv9i5iZd6C7dMGIEm6ifrVtMssdKshrkK0PXPCO%2FweqrwsitTphfa7vpx%2FHUU25v076VSeALw%3D%3D',
    # 'pneumonia_sample2':'',
    # 'pneumonia_sample3':'',
    # 'pneumonia_sample4':'',
    # 'pneumonia_sample5':'',
}
#============================ Model config ==========================
class_dict = {0:'COVID-19',
              1:'NORMAL',
              2:'Viral Pneumonia'}
              
BATCH_SIZE = 64
IMG_SIZE = 224
LR = 0.0001
EPOCHS = 20

LOG_DIR = "output/logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
CHECKPOINT_DIR = 'output/models/snapshots/model-{epoch:03d}-{val_accuracy:03f}.h5'