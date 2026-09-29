# Analiza i Predykcja Rezygnacji Klientów (Customer Churn) w Sklepie E-Commerce

English version: [README.md](README.md)

## Podsumowanie 

Projekt dotyczy analizy wskaźnika rezygnacji klientów (churn rate) w sklepie e-commerce oraz budowy modelu maszynowego przewidującego ryzyko rezygnacji. Celem biznesowym jest identyfikacja kluczowych czynników wpływających na odchodzenie klientów oraz umożliwienie zespołowi retencji wczesnej interwencji.

- Liczba klientów w analizowanej bazie: 3 270 (po wyczyszczeniu)
- Ogólny wskaźnik rezygnacji (Churn Rate): 16,3%
- Wybrany model predykcyjny: XGBoost Classifier (ROC-AUC: 0,9636, Recall: 0,89)

---

## Zmienne

Zbiór danych zawiera informacje o zachowaniach i profilu klientów platformy e-commerce:

- `Tenure`: Czas korzystania z usług firmy przez klienta w miesiącach (numeryczna).
- `WarehouseToHome`: Odległość między magazynem a domem klienta w km (numeryczna).
- `NumberOfDeviceRegistered`: Łączna liczba urządzeń zarejestrowanych przez klienta (numeryczna).
- `PreferedOrderCat`: Preferowana kategoria zamówień klienta w ostatnim miesiącu (kategoryczna).
- `SatisfactionScore`: Ocena satysfakcji klienta z obsługi w skali 1–5 (numeryczna).
- `MaritalStatus`: Stan cywilny klienta (kategoryczna).
- `NumberOfAddress`: Łączna liczba adresów dodanych przez klienta (numeryczna).
- `Complain`: Informacja, czy w ostatnim miesiącu zgłoszono reklamację (binarna: 0 = nie, 1 = tak).
- `DaySinceLastOrder`: Liczba dni od ostatniego zamówienia złożonego przez klienta (numeryczna).
- `CashbackAmount`: Średnia kwota przyznanego zwrotu cashback w ostatnim miesiącu (numeryczna).
- `Churn`: Zmienna docelowa określająca rezygnację klienta (binarna: 0 = pozostał, 1 = zrezygnował).

---

## Architektura projektu

1. Przygotowanie i czyszczenie danych (Python)
2. Eksploracyjna analiza danych (EDA) i analiza korelacji (Python)
3. Stworzenie interaktywnego dashboardu (Power BI)
4. Inżynieria cech i przygotowanie danych do modeli ML (Python)
5. Trening i ewaluacja modeli Random Forest i XGBoost (Python)
6. Wnioski i rekomendacje biznesowe

---

## Wykorzystane biblioteki i narzędzia

* **Analiza i obróbka danych:** `pandas`, `numpy`
* **Wizualizacja danych:** `matplotlib`, `seaborn`
* **Machine Learning & Preprocessing:** `scikit-learn` (`StandardScaler`, `train_test_split`, `RandomForestClassifier`, `metrics`)
* **Zaawansowane modelowanie:** `xgboost` (`XGBClassifier`)
* **Business Intelligence / Raportowanie:** Power BI Desktop

---

## Krok 1: Czyszczenie i przygotowanie danych (Python)

W ramach etapu przygotowania danych wykonano następujące operacje w języku Python:

- Uzupełnienie braków danych: brakujące wartości numeryczne w kolumnach `Tenure`, `WarehouseToHome` oraz `DaySinceLastOrder` uzupełniono medianą wyliczoną dla każdej z tych zmiennych.
- Usuwanie duplikatów: zidentyfikowano i usunięto powtarzające się rekordy w bazie.
- Konwersja typów: poprawiono typy danych dla kolumn numerycznych.
- Weryfikacja zmiennej docelowej: zweryfikowano nierównowagę klas (16,3% Churn vs 83,7% Retained).

---

## Krok 2: Eksploracyjna Analiza Danych (EDA) i korelacje

Przeanalizowano liniowe zależności między cechami a zmienną `Churn` za pomocą korelacji Spearmana:

- `Tenure` (-0,39): najsilniejsza ujemna korelacja, im dłuższy staż klienta w serwisie, tym niższe ryzyko odejścia.
- `Complain` (+0,26): najsilniejszy dodatni predyktor, złożenie reklamacji znacząco podnosi prawdopodobieństwo rezygnacji.
- `CashbackAmount` (-0,17) oraz `DaysSinceLastOrder` (-0,17): wyższe kwoty zwrotu oraz częstsze zakupy umiarkowanie chronią przed churnem.

---

## Krok 3: Dashboard w Power BI

<img width="1177" height="647" alt="image" src="https://github.com/user-attachments/assets/3ef53ea9-a1b5-4449-95a1-f00d5e08b19d" />

Zaprojektowano raport operacyjno-analityczny w Power BI składający się z trzech głównych sekcji:

1. **Wskaźniki KPI**: Liczba Klientów (3 270), Wskaźnik Rezygnacji (Churn Rate) (16,3%), Wskaźnik Reklamacji (28,2%), Średni Staż (10 miesięcy).
2. **Panel Filtrów**: Możliwość segmentacji po stanie cywilnym, stażu oraz kategorii zamówień.
3. **Analiza Szczegółowa**:
   - Wpływ reklamacji: Wskaźnik Rezygnacji (Churn Rate) wynosi 31,8% u klientów ze złożoną reklamacją wobec 10,3% u klientów bez zgłoszeń.
   - Wpływ stażu: 31,2% churnu występuje wśród klientów ze stażem w przedziale 0–6 miesięcy. U klientów ze stażem powyżej 2 lat wskaźnik spada do 0,0%.
   - Zależność satysfakcji i reklamacji: Wysoka ocena satysfakcji (5/5) nie chroni przed odejściem, jeśli towarzyszy jej reklamacja (Churn Rate w tej grupie wynosi 36,5%).
   - Liczba urządzeń: Wskaźnik Rezygnacji rośnie z 7,4% przy 2 zarejestrowanych urządzeniach do 33,7% przy 6 urządzeniach.

---

## Krok 4: Inżynieria cech i przygotowanie do ML

Przed przystąpieniem do modelowania wykonano następujące kroki w Pythonie:

1. Kodowanie zmiennych kategorycznych: Zastosowano One-Hot Encoding (`pd.get_dummies(drop_first=True)`).
2. Podział na zbiór treningowy i testowy: Zastosowano podział 80/20 z zachowaniem proporcji klas (`stratify=y`).
3. Skalowanie cech: Wykorzystano `StandardScaler` do standaryzacji zmiennych numerycznych.

---

## Krok 5: Wyniki modelowania uczenia maszynowego

Do klasyfikacji wytypowano algorytmy drzewiaste. W celu niwelowania nierównowagi klas zastosowano parametry `class_weight='balanced'` (Random Forest) oraz `scale_pos_weight` (XGBoost).

Wyniki na zbiorze testowym (N = 654):

| Metryka | Random Forest | XGBoost (wybrany model) |
| :--- | :---: | :---: |
| ROC-AUC Score | 0,9552 | 0,9636 |
| Recall (Churn = 1) | 0,83 (83%) | 0,89 (89%) |
| Precision (Churn = 1) | 0,70 (70%) | 0,71 (71%) |
| F1-Score (Churn = 1) | 0,76 | 0,79 |
| Accuracy | 0,91 | 0,92 |

Macierz pomyłek dla modelu XGBoost:
- True Negative (poprawnie zaklasyfikowani pozostający): 508
- True Positive (poprawnie wykryty churn): 95 na 107 klientów (89% wykrywalności)
- False Positive (fałszywy alarm): 39
- False Negative (przegapiony churn): 12

---

## Krok 6: Ważność Cech (Feature Importance)

Porównanie ważności cech w modelach pokazało różnice w sposobie podejmowania decyzji przez algorytmy:

- Random Forest: Przypisał najwyższą wagę zmiennym ciągłym: `Tenure` (~28,4%), `CashbackAmount` (~16,1%) oraz `WarehouseToHome` (~10,0%).
- XGBoost: Wskazał jako najważniejsze czynniki `Tenure` (~22,8%) oraz `Complain` (~13,2%), co pokrywa się z wnioskami z fazy EDA i dashboardu Power BI. Istotną rolę odegrały również kategorie produktów o wyższej wartości (Laptopy i akcesoria ~7,8%).

---

## Kluczowe Wnioski 

1. **Ryzyko wczesnego Churnu:**
   - Aż **31,2%** klientów odchodzi w ciągu pierwszych **0–6 miesięcy** korzystania z platformy.
   - Po przekroczeniu progu 2 lat (24 miesięcy) wskaźnik rezygnacji spada niemal do **0%**. Lojalność klientów drastycznie rośnie z czasem.

2. **Reklamacja jako główny "trigger" odejścia:**
   - Klienci, którzy zgłosili reklamację, odchodzą ponad **3-krotnie częściej** (31,8%) w porównaniu do osób bez zgłoszeń (10,3%).
   - Reklamacja wykazuje najsilniejszą dodatnią korelację ze zmienną `Churn` (+0,26).

3. **Paradoks Satysfakcji:**
   - Wysoki poziom deklarowanej satysfakcji (ocena 5/5) **nie chroni klienta przed odejściem**, jeśli w ostatnim czasie złożył on reklamację. W tej grupie wskaźnik rezygnacji osiąga najwyższy poziom **36,5%**.
   - Ocena satysfakcji mierzy ogólne wrażenia, ale nierozwiązany problem operacyjny (reklamacja) natychmiast przeważa nad pozytywną opinią.

4. **Liczba urządzeń:**
   - Wraz ze wzrostem liczby zarejestrowanych urządzeń na koncie wskaźnik churnu rośnie od **7,4%** (dla 2 urządzeń) do aż **33,7%** (dla 6 urządzeń).
   - Może to wskazywać na dzielenie konta z innymi użytkownikami lub problemy z ze spójnością doświadczenia użytkownika (UX) na wielu urządzeniach.

5. **Wpływ programów retencyjnych (Cashback i Zamówienia):**
   - Wyższe kwoty przyznanego cashbacku oraz mniejsza liczba dni od ostatniego zamówienia wykazują ujemną korelację z churnem (~ -0,17). Nagradzanie aktywności skutecznie zatrzymuje klientów w ekosystemie.

---

## Rekomendacje Biznesowe

Na podstawie przeprowadzonej analizy eksploracyjnej oraz wyników modelu predykcyjnego XGBoost wyznaczono 3 główne filary działań retencyjnych:

#### 1. Program Ochrony Nowych Klientów (ochrona w okresie pierwszych 0–6 miesięcy)
* **Problem:** Wskaźnik rezygnacji wśród klientów o stażu do 6 miesięcy wynosi aż **31,2%**, po czym gwałtownie spada w kolejnych miesiącach i latach.
* **Działanie:** 
  * Wdrożenie dedykowanego cyklu onboardingowego (np. seria wiadomości e-mail/push z przewodnikiem po korzyściach płynących z platformy).
  * Przyznawanie spersonalizowanych zachęt i rabatów (**Cashback**) w 2. i 4. miesiącu od rejestracji, aby zbudować nawyk regularnych zakupów.

#### 2. Priorytetyzacja obsługi Reklamacji (SLA & Recovery Program)
* **Problem:** Reklamacja jest najsilniejszym jednostkowym zapalnikiem do odejścia – wskaźnik churnu u osób składających reklamację wynosi **31,8%** (wobec 10,3% u pozostałych). Dodatkowo zidentyfikowano "paradoks satysfakcji", gdzie wysokie oceny (5/5) nie chronią przed odejściem, jeśli towarzyszy im nierozwiązana reklamacja (churn sięga wtedy **36,5%**).
* **Działanie:**
  * Wdrożenie automatycznego flagowania w systemie zgłoszeniowym (Helpdesk/CRM) dla reklamacji pochodzących od nowych klientów (staż < 6 miesięcy).
  * Skrócenie czasu reakcji (SLA) dla tych zgłoszeń do maksymalnie 12–24 godzin oraz automatyczne przyznawanie rekompensaty (np. kod rabatowy lub dodatkowy cashback) po zamknięciu zgłoszenia.

#### 3. Wdrożenie modelu XGBoost do systemu CRM (automatyczne alerty Churn Risk)
* **Problem:** Reagowanie dopiero w momencie, gdy klient przestaje kupować, jest zazwyczaj spóźnione.
* **Działanie:**
  * Integracja wytrenowanego modelu XGBoost z bazą danych klientów w celu cotygodniowego generowania scoringu prawdopodobieństwa odejścia (*Churn Probability*).
  * **Automatyzacja działań retencyjnych:**
    * **Ryzyko wysokie (Prawdopodobieństwo > 70%):** Automatyczne przekazanie rekordu do zespołu Telemarketingu/Retention w celu bezpośredniego kontaktu z indywidualną ofertą.
    * **Ryzyko średnie (Prawdopodobieństwo 40% – 70%):** Automatyczny wyzwalacz (trigger) w systemie marketing automation wysyłający spersonalizowany kupon rabatowy lub propozycję wyższego cashbacku na preferowaną kategorię produktów.
