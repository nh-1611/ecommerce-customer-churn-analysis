# Analiza i Predykcja Rezygnacji Klientów (Customer Churn) w Serwisie E-Commerce

English version: [README.md](README.md)

https://www.kaggle.com/datasets/samuelsemaya/e-commerce-customer-churn/data 

## O projekcie 

Projekt dotyczy analizy historycznej bazy użytkowników serwisu e-commerce ze szczególnym uwzględnieniem wskaźnika rezygnacji oraz budowy modelu uczenia maszynowego przewidującego ryzyko rezygnacji nowych klientów. W celu odzwierciedlenia warunków produkcyjnych wstępna analiza oraz dobór odpowiedniego modelu przebiegał wyłącznie na wstępnie wydzielonych danych treningowych. W przypadku modelu możliwe to było dzięki wykorzystaniu walidacji krzyżowej. Dane testowe traktowane są jako nowa baza klientów, na której przewidywane jest procentowe ryzyko odejścia.
Celem biznesowym jest identyfikacja kluczowych czynników wpływających na decyzję klienta o odejściu, wskazanie grupy zagrożonych użytkowników oraz umożliwienie zespołowi retencji wczesnej interwencji.

---

## W skrócie

**Cel:** identyfikacja czynników rezygnacji klientów e-commerce i wskazanie zagrożonych klientów w nowej bazie, w celu umożliwienia zespołowi retencji wczesnej interwencji.

**Dane:** publiczny zbiór z Kaggle, 3 941 obserwacji (po czyszczeniu: 2 712 w zbiorze treningowym, 768 w testowym). Rezygnacja w danych historycznych: **16,3%**. Zbiór testowy służy jako symulacja nowej bazy klientów bez znanych decyzji.

**Podejście:** analiza statystyczna (testy hipotez z siłą efektu) → 5-krotna stratyfikowana walidacja krzyżowa w pipeline'ie (SMOTE i skalowanie wewnątrz foldów) → Random Forest po strojeniu → interpretacja (SHAP, ważność cech, regresja logistyczna jako baseline) → predykcja dla nowej bazy → dwa dashboardy w Power BI.

**Model:** Random Forest (wybrany spośród regresji logistycznej, Random Forest i XGBoost).

| Metryka (walidacja krzyżowa, zbiór treningowy) | Wartość |
| :--- | :---: |
| ROC-AUC | 0,93 |
| Recall (churn = 1) | 0,73 |
| Precision (churn = 1) | 0,68 |
| F1 (churn = 1) | 0,71 |

Próg decyzyjny: **0,35**

**Predykcja dla nowej bazy:** **165 z 768 klientów (21,5%)** powyżej progu ryzyka, w tym 75 w grupie krytycznej (75-100%).

**Najważniejsze wnioski:**
- **Staż jest głównym czynnikiem:** w pierwszych 0-6 miesiącach odchodzi 31,5% klientów, a po 2 latach praktycznie żaden (r_rb = -0,60, duży efekt; najsilniejszy wynik z testów).
- **Reklamacje:** churn 31,0% u klientów ze skargą vs 10,5% bez skargi.
- **Liczba urządzeń:** churn rośnie z 6,4% (2 urządzenia) do 35,2% (6 urządzeń).
- **Cechy według SHAP:** staż, stan cywilny (singiel), preferowana kategoria (telefony), cashback.
- **Zaskakujące wyniki:** wyższa satysfakcja i mniejsza liczba dni od ostatniego zamówienia wiążą się z częstszym odejściem, co prawdopodobnie wynika ze specyfiki zbioru danych.

**Zawartość repozytorium:**
- `ecommerce_churn_analysis.ipynb`: czyszczenie, EDA, testy hipotez
- `ecommerce_churn_model.ipynb`: modelowanie, interpretacja, predykcja
- `ecommerce_churn_analysis_dashboard_pl.pbix`: dashboard analityczny w Power BI
- `ecommerce_churn_prediction_pl.pbix`: dashboard predykcyjny w Power BI
  
---

## Zmienne

3 941 obserwacji przed czyszczeniem

Zbiór danych zawiera informacje o zachowaniach i profilu klientów platformy e-commerce:

- `Tenure`: Czas korzystania z usług firmy przez klienta w miesiącach (numeryczna).
- `WarehouseToHome`: Odległość między magazynem a domem klienta w km (numeryczna).
- `NumberOfDeviceRegistered`: Łączna liczba urządzeń zarejestrowanych przez klienta (numeryczna).
- `PreferedOrderCat`: Preferowana kategoria zamówień klienta w ostatnim miesiącu (kategoryczna).
- `SatisfactionScore`: Ocena satysfakcji klienta z obsługi w skali 1-5 (numeryczna).
- `MaritalStatus`: Stan cywilny klienta (kategoryczna).
- `NumberOfAddress`: Łączna liczba adresów dodanych przez klienta (numeryczna).
- `Complain`: Informacja, czy w ostatnim miesiącu zgłoszono reklamację (binarna: 0 = nie, 1 = tak).
- `DaySinceLastOrder`: Liczba dni od ostatniego zamówienia złożonego przez klienta (numeryczna).
- `CashbackAmount`: Średnia kwota przyznanego zwrotu cashback w ostatnim miesiącu (numeryczna).
- `Churn`: Zmienna docelowa określająca rezygnację klienta (binarna: 0 = pozostał, 1 = zrezygnował).

---

## Architektura projektu

1. Przygotowanie, wstępny podział i czyszczenie danych (Python)
2. Eksploracyjna analiza danych (EDA), badanie normalności, analiza korelacji, badanie współliniowości, testowanie hipotez (Python)
3. Stworzenie interaktywnego dashboardu związanego z EDA (Power BI)
4. Inżynieria cech i przygotowanie danych do modeli ML (Python)
5. Trening i ewaluacja modeli regresji logistycznej, Random Forest i XGBoost (Python)
6. Interpretacja ważności cech (Python)
7. Predykcja ryzyka odejścia nowej bazy klientów (Python)
8. Stworzenie interaktywnego dashboardu związanego z predykcją (Power BI)
9. Wnioski i rekomendacje biznesowe
10. Ograniczenia i dalsze kroki

---

## Wykorzystane narzędzia

* **Analiza i obróbka danych:** `pandas`, `numpy`, `scipy` (`stats`), `statsmodels`
  
* **Wizualizacja danych:** `matplotlib`, `seaborn`

* **Modele uczenia maszynowego**
* `scikit-learn` - podział danych (`train_test_split`), skalowanie (`StandardScaler`), optymalizacja parametrów (`GridSearchCV`), walidacja krzyżowa (`StratifiedKFold`, `cross_validate`) oraz ocena modeli (`classification_report`, `confusion_matrix`, `roc_auc_score`), regresja logistyczna (`LogisticRegression`), las losowy (`RandomForestClassifier`)
* `imblearn` - obsługa nierównowagi klas (`SMOTE`), budowa potoków przetwarzania (`Pipeline`)
* `xgboost` (`XGBClassifier`)
  
* **Business Intelligence / Raportowanie:** Power BI Desktop

---


## Krok 1: Podział na dane treningowe i testowe 

Dane podzielono na dane treningowe (3 152) i testowe (789) **PRZED analizą**, aby odzwierciedlić realia produkcyjne. Takie działanie ma na celu przede wszystkim zapobiegnięcie **wyciekowi danych** (*Data Leakage*). Podział surowych danych przed imputacją braków, skalowaniem czy kodowaniem gwarantuje, że statystyki (np. mediany) są obliczane wyłącznie na zbiorze treningowym.
Zastosowano **podział 80/20 z zachowaniem proporcji klas**, stratyfikacja zapewnia zachowanie identycznych proporcji klas (churn vs brak churnu) w obu zbiorach, eliminując ryzyko losowego skrzywienia próby.

Dane treningowe postrzegane są jako historyczna baza klientów, natomiast dane testowe mają za zadanie odzwierciedlać nową bazę klientów, dla której przewidywane są prawdopodobieństwa podjęcia decyzji o rezygnacji z serwisu.

## Krok 2: Czyszczenie i przygotowanie danych 

W ramach etapu przygotowania danych wykonano następujące operacje:

- **Uzupełnienie braków danych:** brakujące wartości numeryczne dla zbioru treningowego i zbioru testowego w kolumnach `Tenure`, `WarehouseToHome` oraz `DaySinceLastOrder` uzupełniono medianą wyliczoną dla każdej z tych zmiennych na danych ze zbioru treningowego.
- **Usuwanie duplikatów:** zidentyfikowano i usunięto powtarzające się rekordy w bazie.
- **Konwersja typów:** poprawiono typy danych dla kolumn numerycznych.
- **Przycinanie wartości odstających:** za pomocą zakresu międzykwartylowego (IQR) wyznaczono dolne i górne granice dla każdej zmiennej, a następnie bazując na nich przycięto wartości odstające. Metoda *cappingu* ma na celu ograniczenie wpływu skrajnych wartości, jednocześnie zachowując informacje o danych. Na zbiorze testowym wykorzystano granice obliczone na danych ze zbioru treningowego.
- **Mapowanie wartości:** ujednolicono błędnie opisane rekordy.

Po wyczyszczeniu zbiór treningowy liczy 2712 obserwacje, natomiast zbiór testowy 768.

---

## Krok 2: Eksploracyjna Analiza Danych (EDA)

Przeanalizowano statystyki opisowe dla danych treningowych.

> * Wyczyszczona historyczna baza klientów ma **2712 rekordy.**
> * Baza klientów jest niezrównoważona. Z serwisu odchodzi około **16,3%** użytkowników.
> * Zdecydowana większość cech ciągłych (`Tenure`, `WarehouseToHome`, `NumberOfAddress`, `DaySinceLastOrder`, `CashbackAmount`) wykazuje dodatnią skośność (0,69-0,95). Wartości średnie są zawyżane przez długie prawostronne ogony. Podejrzewany jest brak normalności rozkładu.
> * Wysokie odchylenie standardowe wskazują na rozbieżność w doświadczeniach klientów, co stanowi kluczowy punkt wyjścia do badania zależności cech z rezygnacją z usług.

> * Zdecydowanie najliczniejszą grupę stanowią klienci o stażu 0-1 miesiąca. **Ponad 500 użytkowników to zupełnie nowi klienci.**
> * Większość klientów ma zarejestrowane **3 lub 4 urządzenia**.
> * Sprzedaż jest zdecydowanie zdominowana przez dwie kategorie: **laptopy i akcesoria (37,0%, 1 004)** oraz **telefony komórkowe (36,1%, 980)**, które łącznie odpowiadają za **73,1%** wszystkich zamówień.
> * Najliczniejszą grupą są klienci wskazujący **ocenę neutralną, czyli 3 (28,9%, 783).** Najniższą ocenę 1 wskazało 21,6% (586), podczas gdy najwyższe oceny (4 i 5) stanowią łącznie 37,7% (1 022).
> * Ponad połowę klientów stanowią osoby **w związku małżeńskim (51,7%, 1 401)**. Osoby samotne stanowią 31,7% (859), a rozwiedzione 16,7% (452).
> * Większość użytkowników posiada **od 2 do 3 zapisanych adresów dostawy** (ponad 1 200 klientów).
> * Większość klientów (71,7%, 1 945) nie złożyła żadnej reklamacji. Z kolei **28,3% (767) klientów zgłosiło przynajmniej jedną skargę**. Odsetek ponad 28% skarg w bazie jest stosunkowo wysoki i wskazuje na potencjalne problemy z obsługą klienta, dostawą lub jakością produktów.
> * W przypadku **zwrotu typu cashback**, największe zagęszczenie klientów znajduje się wokół **wartości 150 (140-160)**.

---

## Krok 3: Testowanie normalności

Sprawdzono normalność rozkładu, aby w następnych krokach móc dobrać odpowiednie testy do dalszej analizy danych. Przez wzgląd na liczebność obserwacji skorzystano z testu D'Agostino-Pearsona opartego na skośności i kurtozie. Wyniki poparto wykresem kwantyl-kwantyl. Dla każdej zmienna **hipoteza o rozkładzie normalnym została odrzucona.**

---

## Krok 4: Korelacje

Do badania korelacji wykorzystano metodę Spearmana, aby obejść problem braku normalności rozkładu. 

>* Najsilniejszą zależność z rezygnacją wykazuje **staż klienta** (*r* = -0,39), który charakteryzuje się ujemną korelacją. Klienci o długim stażu rezygnują znacznie rzadziej, co sugeruje, >że **krytycznym okresem mogą być pierwsze miesiące po rejestracji.**
>* Kolejnym głównym czynnikiem ryzyka mogą być **reklamacje** (*r* = 0,25). Problemy z obsługą lub produktem są powiązane z decyzją o odejściu.
>* **Wyższa kwota zwrotu wiąże się z niższą skłonnością do odejścia** (*r* = -0,17) - przypuszczalnie programy lojalnościowe skutecznie zatrzymują klientów.
>* Co ciekawe, **dłuższy czas od ostatniego zamówienia wiąże się z pozostawaniem ludzi w serwisie (*r* = -0,16).**
>* Liczba zarejestrowanych urządzeń (*r* = 0,11) oraz ocena satysfakcji (*r* = 0,10) wykazują słabą dodatnią korelację z odejściem z serwisu.
>* Dystans do magazynu (*r* = 0,06) i liczba zapisanych adresów (*r* = 0,03) wykazują znikomą zależność z rezygnacją.

>* Żadna para zmiennych niezależnych nie przekracza progu korelacji *r* > 0,40. Oznacza to **brak problemu nadmiarowości danych i brak konieczności usuwania cech ze względu na współliniowość**. Dodatkowo, potwierdza to analiza VIF.

---

## Krok 5: Testowanie hipotez

Dla zmiennych ciągłych i porządkowych wykorzystano test U-Manna Whitneya, poparty siłą efektu. Dla zmiennych kategorycznych wykorzystano test Chi-Square. Analizę wspomogły wykresy.

| Hipoteza | Zmienna | Test | Statystyka | p-value | Siła efektu | Interpretacja siły | Wniosek |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| H1 | `Tenure` | U Manna-Whitneya (jednostronny) | U = 803 668 | < 0,001 | r_rb = -0,60 | duży | Odchodzący klienci mają istotnie krótszy staż |
| H2 | `WarehouseToHome` | U Manna-Whitneya (jednostronny) | U = 451 246 | < 0,001 | r_rb = 0,10 | mały | Odchodzący mieszkają dalej od magazynu, ale efekt jest słaby |
| H3 | `NumberOfDeviceRegistered` | U Manna-Whitneya (jednostronny) | U = 419 429 | ≈ 1,000 | r_rb = 0,16 | mały | Hipoteza odrzucona: więcej urządzeń mają odchodzący |
| H3 (dodatkowo) | `NumberOfDeviceRegistered` | Chi² | χ² = 44,84 | < 0,001 | V Craméra = 0,13 | mały | Istotna zależność, w kierunku przeciwnym do założonego |
| H4 | `PreferedOrderCat` | Chi² | χ² = 149,25 | < 0,001 | V Craméra = 0,23 | mały | Kategoria istotnie związana z churnem |
| H5 | `SatisfactionScore` | U Manna-Whitneya (jednostronny) | U = 421 993 | ≈ 1,000 | r_rb = 0,16 | mały | Hipoteza odrzucona: wyższa satysfakcja nie chroni przed odejściem |
| H5 (dodatkowo) | `SatisfactionScore` | Chi² | χ² = 35,61 | < 0,001 | V Craméra = 0,11 | mały | Istotna zależność, kierunek nieintuicyjny |
| H6 | `MaritalStatus` | Chi² | χ² = 99,73 | < 0,001 | V Craméra = 0,19 | mały | Istotna zależność; osoby w związku małżeńskim odchodzą rzadziej |
| H7 | `NumberOfAddress` | U Manna-Whitneya (jednostronny) | U = 482 341 | 0,904 | r_rb = 0,04 | znikomy | Brak istotnej różnicy; liczba adresów nie wiąże się z lojalnością |
| H8 | `Complain` | Chi² | χ² = 168,64 | < 0,001 | V Craméra (= φ) = 0,25 | mały | Klienci z reklamacją odchodzą istotnie częściej |
| H9 | `DaySinceLastOrder` | U Manna-Whitneya (jednostronny) | U = 628 801,5 | ≈ 1,000 | r_rb = -0,25 | mały | Hipoteza odrzucona: odchodzący mają *mniej* dni od ostatniego zamówienia |
| H10 | `CashbackAmount` | U Manna-Whitneya (jednostronny) | U = 633 353,5 | < 0,001 | r_rb = -0,26 | mały | Odchodzący mają istotnie niższy cashback |

* Poziom istotności α = 0,05.

* H1: Klienci o krótszym stażu (`Tenure`) w serwisie odchodzą częściej niż klienci długoterminowi.
> * U nowych użytkowników istnieje istotnie większe ryzyko rezygnacji niż u klientów o dłuższym stażu. Sugerowane jest wzmocnienie działań onboardingowych i retencyjnych we wczesnej fazie cyklu życia klienta.
  
* H2: Większa odległość magazynu od adresu dostawy klienta (`WarehouseToHome`) wiąże się ze wzrostem wskaźnika odejść.
> * Odległość magazynu od domu jest istotnie wyższa u klientów odchodzących niż u tych, którzy pozostają w serwisie. Zależność nie jest silna, lecz potrzebne może być wdrożenie działań w tym zakresie.
 
* H3: Większa liczba zarejestrowanych urządzeń (`NumberOfRegisteredDevice`) świadczy o przywiązaniu użytkownika do serwisu, co zmniejsza ryzyko rezygnacji w tej grupie.
> * Podobnie jak z liczbą adresów, liczba zarejestrowanych urządzeń nie przekłada się na lojalność do serwisu. Wśród klientów o największej liczbie połączonych urządzeń znajduje się największy odsetek odchodzących klientów.

* H4: Klienci, których preferowaną kategorią (`PreferedOrderCat`) jest elektronika charakteryzują się innymi zachowaniami związanymi z odejściem w porównaniu do klientów wybierających bardziej codzienne kategorie (żywność, moda).
> * Najchętniej wybierana przez klienta kategoria ma istotny wpływ na decyzję o rezygnacji. Osoby kupujący głównie żywność niemal nie rezygnują z serwisu, czego nie można powiedzieć o użytkownikach wybierających telefony komórkowe.

* H5: Niższa ocena satysfakcji klienta (`SatisfactionScore`) wiąże się z większym prawdopodobieństwem odejścia.
> * Wysoka ocena satysfakcji nie zatrzymuje klienta w serwisie. Użytkownicy lepiej oceniający serwis są jednocześnie tymi, którzy zgłaszają skargi, wpływające na ich decyzję o rezygnacji.

* H6: Klienci w związku małżeńskim (`MaritalStatus`) odchodzą rzadziej niż klienci o innym statusie związku.
> * Status związku ma istotny wpływ na zjawisko rezygnacji wśród klientów. Użytkownicy będący w związkach małżeńskich wykazują większą stabilność i lojalność wobec serwisu, podczas gdy singielki i single wymagają dedykowanych strategii retencyjnych.

* H7: Większa liczba zapisanych adresów (`NumberOfAddress`) świadczy o przywiązaniu użytkownika do serwisu, co zmniejsza ryzyko rezygnacji w tej grupie.
> * Liczba zapisanych w serwisie adresów nie jest wyższa u klientów, którzy nie decydują się na rezygnację. Oznacza to, że nie wiąże się to realnie ze stabilnością i lojalnością użytkowników, a prawdopodobnie wynika wyłącznie z wygody.

* H8: Klienci zgłaszający skargi (`Complain`) mają istotnie wyższy wskaźnik rezygnacji niż klienci, którzy ich nie zgłaszają.
> * Zachowania związane z rezygnacją istotnie różnią się między klientami zgłaszającymi skargi a klientami, którzy nigdy tego nie zrobili. Siła tego związku jest niewielka, lecz istotna. Złoszenie reklamacji jest silnym sygnałem ostrzegawczym. Należy wdrożyć działania dedykowane sprawnemu rozpatrywaniu skarg w celu zatrzymania użytkownika.

* H9: Dłuższy czas, jaki upłynął od ostatniego zamówienia (`DaySinceLastOrder`), zwiększa prawdopodobieństwo, że klient zrezygnuje z serwisu.
> * Wbrew oczekiwaniom, liczba dni od ostatniego zamówienia jest taka sama lub niższa w grupie klientów, którzy zrezygnowali z serwisu niż w grupie, która pozostała. Taki wynik podkreśla, że aktywność nie gwarantuje zatrzymania użytkownika.

* H10: Wyższy zwrot (`CashbackAmount`) skutecznie zatrzymuje użytkownika w serwisie.
> * Odchodzący klienci charakteryzują się istotnie niższą sumą zwrotu cashback niż użytkownicy pozostający w serwisie. Pozwala to na sformułowanie wniosku, mówiącego że cashback i częste zakupy kształtują lojalność klienta do platformy. W związku z tym, warto rozważyć więcej działań mających na celu zwiększenie przywiązania użytkownika do serwisu.

---

## Krok 6: Dashboard analityczny w Power BI

<img width="1302" height="707" alt="ecommerce_churn_analysis_dashboard_pl" src="https://github.com/user-attachments/assets/6a765bd0-49b4-40fc-bc91-51fdf5272784" />

Zaprojektowano raport operacyjno-analityczny w Power BI składający się z trzech głównych sekcji:

1. **Wskaźniki KPI**: Liczba Klientów (2 712), Wskaźnik Rezygnacji (Churn Rate) (16,3%), Wskaźnik Reklamacji (28,3%), Średni Staż (10 miesięcy), Wskaźnik Satysfakcji (3,0).
2. **Panel filtrów**: Możliwość segmentacji po stanie cywilnym, stażu według grup oraz preferowanej kategorii zamówień.
3. **Analiza szczegółowa**:
   - Wpływ reklamacji: Wskaźnik Rezygnacji (Churn Rate) wynosi 31,0% u klientów ze złożoną reklamacją wobec 10,5% u klientów bez zgłoszeń.
   - Wpływ stażu: 31,5% churnu występuje wśród klientów ze stażem w przedziale 0-6 miesięcy. U klientów ze stażem powyżej 2 lat wskaźnik spada do 0%.
   - Liczba urządzeń: Wskaźnik Rezygnacji rośnie z 6,4% przy 2 zarejestrowanych urządzeniach do 35,2% przy 6 urządzeniach.

---

## Krok 7: Wybór modelu uczenia maszynowego

Przed przystąpieniem do modelowania zakodowano zmienne kategoryczne za pomocą One-Hot Encoding.

Poprzez stratyfikowaną 5-krotną walidację krzyżową, zbiór treningowy podzielono na 5 równych części z zachowaniem proporcji klasy mniejszościowej (odchodzących klientów). Zapobiega to niestabilności ocen metryk w poszczególnych pętlach. Aby całkowicie wyeliminować ryzyko przecieku danych, wszystkie kroki transformacji, w tym standaryzacja cech numerycznych oraz generowanie sztucznych próbek SMOTE są wykonywane wyłącznie wewnątrz pętli (*pipeline*) walidacji krzyżowej na danych treningowych danego folda.

| Metryka | Regresja logistyczna | Random Forest | XGBoost |
| :--- | :---: | :---: | :---: |
| ROC-AUC Score | 0,8819 | 0,9282 | 0,9240 |
| Recall (Churn = 1) | 0,8053 | 0,7194 | 0,6879 |
| Precision (Churn = 1) | 0,4350 | 0,6750 | 0,7009|
| F1-Score (Churn = 1) | 0,56 | 0,70 | 0,69 |

Porównano ze sobą 3 modele:
* **Regresja logistyczna** jako najprostszy z modeli służy jako baseline dla algorytmów drzewiastych. Osiąga najwyższy wskaźnik *Recall* (80,53%), co oznacza, że wyłapuje najwięcej odchodzących klientów. Wykazuje jednak bardzo niską *Precision* (43,50%) oraz najniższy *F1-Score* (0,56). W praktyce generuje bardzo dużo fałszywych alarmów, klasyfikując stałych klientów jako zagrożonych odejściem, co podnosi koszty kampanii retencyjnych.
* **XGBoost** osiąga najwyższą precyzję (70,09%), tworząc najbardziej trafne trafienia, ale odbywa się to kosztem niższego wskaźnika *Recall* (68,79%), czyli przeoczenia większej liczby rzeczywistych odejść. Dodatkowo, Osiąga bardzo wysoki *ROC-AUC* (0,92) oraz *F1-Score* (0,69).
* **Las losowy (*Random Forest*)** dominuje w głównej metryce oceny jakości ogólnej: *ROC-AUC* (0,93) oraz *F1-Score* (0,70). Utrzymuje najbardziej optymalny balans pomiędzy wykrywaniem odejść (*Recall* = 71,94%) a unikaniem fałszywych alarmów (*Precision* = 67,50%).

Do etapu strojenia hiperparametrów wybrany został ***Random Forest***. W kontekście przeciwdziałania rezygnacjom klientów zależało nam na zminimalizowaniu zarówno kosztów niepodjęcia działań wobec odchodzącego klienta (braki w *Recall*), jak i kosztów niepotrzebnych rabatów wysyłanych do stałych klientów (braki w *Precision*). *Random Forest* zapewnia **optymalny kompromis (najwyższy *F1-Score*)**. Osiągnięcie najwyższego wyniku *ROC-AUC* świadczy o doskonałej zdolności separacji klas przez model niezależnie od przyjętego progu odcięcia, co stanowi idealny fundament pod późniejszy tuning i dostrajanie progu decyzyjnego.

Macierz pomyłek dla modelu *Random Forest*:
- True Negative (poprawnie zaklasyfikowani pozostający): 2116
- True Positive (poprawnie wykryty churn): 319
- False Positive (fałszywy alarm): 154
- False Negative (przegapiony churn): 124

---

## Krok 8: Szukanie najlepszych parametrów

Zidentyfikowano najlepsze parametry dla *Random Forest*:

Kryterium podziału: entropia okazała się nieco lepsza od standardowego Giniego w budowaniu bardziej czystych podziałów w drzewach.
Maksymalna głębokość drzewa: drzewa mogą rosnąć bez sztucznego ograniczenia głębokości, dopasowując się do złożonych relacji w danych.
Minimalna liczba próbek w liściu: 1 oraz w węźle: 2.
Liczba drzew w lesie: las składa się z 200 drzew decyzyjnych, co zapewnia stabilność i wysoką moc predykcyjną.

**Ocena skuteczności tuningu**

Uzyskany najlepszy *F1-Score* wynosi 0,71 w porównaniu do wstępnego modelu bazowego, który miał 0,70. Wartość *Precision* wzrosła z 0,67 do 0,68, a *Recall* z 0,72 do 0,73. Dostrajanie hiperparametrów przyniosło delikatną poprawę jakości wykrywania churnu.

Macierz pomyłek dla modelu *Random Forest* o polepszonych parametrach:
- True Negative (poprawnie zaklasyfikowani pozostający): 2120
- True Positive (poprawnie wykryty churn): 323
- False Positive (fałszywy alarm): 150
- False Negative (przegapiony churn): 119

## Krok 9: Interpretacja
* **Ważność cech:** Staż jest zdecydowanie najważniejszą cechą (waga powyżej 0,25). Oznacza to, że staż klienta w firmie ma fundamentalne znaczenie dla przewidywania ryzyka jego odejścia.
Wysokość cashbacku oraz liczba dni od ostatniego zamówienia zajmują kolejno drugie i trzecie miejsce, co wskazuje, że aktywność zakupowa oraz profity finansowe mocno wiążą się z lojalnością.
Odległość magazynu od domu oraz status cywilny wpływają na logistykę i profil demograficzny klienta.
* **Interpretacja regresji logistycznej:** Złożenie skargi przez klienta zwiększa szansę jego odejścia aż o 109,2%. Specyficzna kategoria zamawianych produktów (Others) podnosi ryzyko odejścia o około 79,5%. Większa liczba adresów powiązanych z kontem zwiększa ryzyko podjęcia decyzji o rezygnacji o 74,7%, a większa liczba zarejestrowanych urządzeń powoduje wzrost o ok. 59,9%. Status singla napędza prawdopodobieństwo rezygnacji o 48,7%. Dni od ostatniego zamówienia wykazuje ujemny wpływ na poziomie -23,8%, co oznacza, że świeża i regularna aktywność zakupowa obniża ryzyko odejścia.
* **SHAP:** Wysokie wartości stażu skupiają się po lewej stronie (ujemne wartości SHAP), co oznacza, że duży staż klienta silnie chroni przed odejściem. Niskie wartości stażu (kolor niebieski) przesunięte są w prawo, co potwierdza, że nowi klienci są najbardziej podatni na rezygnację. Wysokie wartości (czerwone punkty) dla bycia singlem oraz preferowania zamówień przez telefon generują dodatnie wartości SHAP, co zwiększa ryzyko churnu. Zróżnicowany rozkład punktów przy wysokości cashbacku pokazuje, że umiarkowane i wyższe kwoty cashbacku pomagają zatrzymać klientów (przesunięcie w lewo). Zgodnie z wcześniejszymi modelami, obecność skarg mocno przesuwa wartości w stronę dodatnią (prawa strona wykresu), co drastycznie podnosi prawdopodobieństwo rezygnacji.

---

## Krok 10: Prognoza ryzyka dla nowej bazy klientów i dashboard w Power BI

Przyjęty model zastosowano na danych testowych w celu obliczenia ryzyka odejścia oraz wskazania zagrożonych klientów. Wyniki **nie zostały** skalibrowane, przez co nie odzwierciedlają, a w większości przypadków wręcz zawyżają prawdopodobieństwa. Na potrzeby projektu potraktowano je jednak jako prawdopodobieństwo rezygnacji.

Przyjęto próg = 0,35, aby wyłonić jak najwięcej użytkowników potencjalnie skłonnych do rezygnacji z serwisu.

<img width="1322" height="738" alt="image" src="https://github.com/user-attachments/assets/3beb2245-c6f7-404b-a6ac-6599077c2deb" />

1. **Wskaźniki:** Wśród 768 klientów, aż **165 jest zagrożonych odejściem**, co przekłada się na 21,5% całej bazy. Średnie Prawdopodobieństwo rezygnacji wśród osób zagrożonych wynosi **71,6%.**
2. **Prawdopodobieństwo rezygnacji:** Ze 165 zagrożonych użytkowników, aż 75 znajduje się w grupie krytycznego ryzyka (75%-100%).
3. **Panel filtrów**: Możliwość segmentacji po stanie cywilnym, stażu według grup, preferowanej kategorii zamówień, złożeniu reklamacji oraz ocenie satysfakcji.

---

## Kluczowe wnioski 

1. **Ryzyko wczesnej rezygnacji:**
   - Aż **31,5%** klientów odchodzi w ciągu pierwszych **0-6 miesięcy** korzystania z platformy.
   - Po przekroczeniu progu 2 lat (24 miesięcy) wskaźnik rezygnacji spada niemal do **0%**. Lojalność klientów drastycznie rośnie z czasem.

2. **Reklamacja jako główny *trigger* odejścia:**
   - Klienci, którzy zgłosili reklamację, odchodzą ponad **3-krotnie częściej** (31,0%) w porównaniu do osób bez zgłoszeń (10,5%).
   - Reklamacja wykazuje najsilniejszą dodatnią korelację ze zmienną `Churn` (+0,25).

3. **Paradoks satysfakcji:**
   - Wysoki poziom deklarowanej satysfakcji (ocena 5/5) **nie chroni klienta przed odejściem**, jeśli w ostatnim czasie złożył on reklamację. Ocena satysfakcji mierzy ogólne wrażenia, ale nierozwiązany problem operacyjny (reklamacja) natychmiast przeważa nad pozytywną opinią.

4. **Liczba urządzeń:**
   - Wraz ze wzrostem liczby zarejestrowanych urządzeń na koncie wskaźnik churnu rośnie od **6,4%** (dla 2 urządzeń) do aż **35,2%** (dla 6 urządzeń). Może to wskazywać na dzielenie konta z innymi użytkownikami lub problemy z ze spójnością doświadczenia użytkownika (UX) na wielu urządzeniach.

5. **Wpływ programów retencyjnych (cashback i zamówienia):**
   - Wyższe kwoty przyznanego cashbacku oraz mniejsza liczba dni od ostatniego zamówienia wykazują ujemną korelację z churnem (~ -0,17). Nagradzanie aktywności skutecznie zatrzymuje klientów w systemie.

---

## Rekomendacje biznesowe

Na podstawie przeprowadzonej analizy eksploracyjnej oraz wyników modelu predykcyjnego Random Forest wyznaczono 3 główne filary działań retencyjnych:

#### 1. Program ochrony nowych klientów (ochrona w okresie pierwszych 0-6 miesięcy)
* **Problem:** Wskaźnik rezygnacji wśród klientów o stażu do 6 miesięcy wynosi aż **31,5%**, po czym gwałtownie spada w kolejnych miesiącach i latach.
* **Działanie:** 
  * Wdrożenie dedykowanego cyklu onboardingowego (np. seria wiadomości e-mail/push z przewodnikiem po korzyściach płynących z platformy).
  * Przyznawanie spersonalizowanych zachęt i rabatów (**cashback**) w 2. i 4. miesiącu od rejestracji, aby zbudować nawyk regularnych zakupów.

#### 2. Priorytetyzacja obsługi reklamacji (SLA & Recovery Program)
* **Problem:** Reklamacja jest najsilniejszym jednostkowym zapalnikiem do odejścia - wskaźnik churnu u osób składających reklamację wynosi **31,0%** (wobec 10,5% u pozostałych). Dodatkowo zidentyfikowano "paradoks satysfakcji", gdzie wysokie oceny (5/5) nie chronią przed odejściem, jeśli towarzyszy im nierozwiązana reklamacja (churn sięga wtedy **35,8%**).
* **Działanie:**
  * Wdrożenie automatycznego flagowania w systemie zgłoszeniowym (Helpdesk/CRM) dla reklamacji pochodzących od nowych klientów (staż < 6 miesięcy).
  * Skrócenie czasu reakcji (SLA) dla tych zgłoszeń do maksymalnie 12-24 godzin oraz automatyczne przyznawanie rekompensaty (np. kod rabatowy lub dodatkowy cashback) po zamknięciu zgłoszenia.

#### 3. Wdrożenie modelu Random Forest do systemu CRM (automatyczne alerty Churn Risk)
* **Problem:** Reagowanie dopiero w momencie, gdy klient przestaje kupować, jest zazwyczaj spóźnione.
* **Działanie:**
  * Integracja wytrenowanego modelu Random Forest z bazą danych klientów w celu cotygodniowego generowania scoringu prawdopodobieństwa odejścia (*Churn Probability*).
  * **Automatyzacja działań retencyjnych:**
    * **Ryzyko wysokie (Prawdopodobieństwo > 75%):** Automatyczne przekazanie rekordu do zespołu marketingu/retencji w celu bezpośredniego kontaktu z indywidualną ofertą.
    * **Ryzyko średnie (Prawdopodobieństwo 35%-75%):** Automatyczny wyzwalacz w systemie wysyłający spersonalizowany kupon rabatowy lub propozycję wyższego cashbacku na preferowaną kategorię produktów.
   
---

## Ograniczenia

1. **Capping i imputacja przed walidacją krzyżową:** Granice cappingu (IQR) oraz mediany do uzupełniania braków wyznaczono na całym zbiorze treningowym przed walidacją krzyżową, a nie wewnątrz pipeline'u. Powoduje to niewielki wyciek informacji między foldami. Zbioru testowego to nie dotyczy, ponieważ użyto w nim wartości wyliczonych wyłącznie na zbiorze treningowym.

2. **Brak kalibracji prawdopodobieństw:** Wyniki modelu są w projekcie traktowane jako prawdopodobieństwa rezygnacji, jednak nie przeprowadzono ich kalibracji, a model uczono z użyciem SMOTE. Rozkład wyników jest silnie spolaryzowany. Należy je więc interpretować przede wszystkim jako ranking ryzyka klientów, a nie jako rzeczywiste szanse odejścia.

3. **Próg decyzyjny:** Przyjęty próg 0,35 został dobrany na predykcjach z walidacji krzyżowej (out-of-fold) i uzasadniony kosztem działań retencyjnych. Nie przeprowadzono jednak szczegółowej analizy kosztowej (koszt akcji retencyjnej vs wartość utraconego klienta) przez brak takich danych.

4. **Brak oceny modelu na nowej bazie klientów:** Zbiór testowy nie zawiera informacji o rzeczywistej decyzji klientów i służy jako symulacja nowej bazy. Dashboard predykcyjny pokazuje przewidywania, a nie zmierzoną skuteczność. Jakość modelu oszacowano wyłącznie na podstawie walidacji krzyżowej na zbiorze treningowym. W rzeczywistym wdrożeniu konieczny byłby monitoring (porównanie predykcji z faktycznymi rezygnacjami).

5. **Zależności, nie przyczyny:** Analiza korelacji, testów hipotez, współczynników regresji i wartości SHAP pokazuje związki między cechami a rezygnacją, a nie relacje przyczynowe. Część wyników jest nieintuicyjna (np. wyższa satysfakcja i mniejsza liczba dni od ostatniego zamówienia wiążą się z częstszym odejściem) i prawdopodobnie wynika ze specyfiki zbioru danych, a nie z rzeczywistych zachowań klientów.

6. **Zbiór danych:** Projekt oparto na jednym, publicznym zbiorze danych (Kaggle) o ograniczonej liczebności (3 941 obserwacji przed czyszczeniem). Zbiór nie zawiera identyfikatora klienta, więc usuwanie duplikatów opiera się na identyczności wszystkich cech. Imiona i nazwiska na dashboardzie predykcyjnym są syntetyczne.

7. **Strojenie hiperparametrów:** Poprawa po strojeniu Random Forest była niewielka (poprawa F1-Score z 0,70 na 0,71) i mieści się w zakresie zmienności między foldami, więc nie należy jej interpretować jako istotnej różnicy.

8. **Wielokrotne testowanie hipotez:** Przy 10 testach nie zastosowano korekty na wielokrotne testowanie.

---

## Dalsze kroki

- Ocena krzywej kalibracji na predykcjach out-of-fold i zastosowanie kalibracji, aby wyniki mogły być interpretowane jako prawdopodobieństwa.
- Przeniesienie cappingu i imputacji do pipeline'u walidacji krzyżowej.
- Dobór progu na podstawie analizy kosztowej i tabeli recall/precision dla różnych progów.
- Dodanie na dashboardzie głównych czynników ryzyka dla każdego klienta.
- Monitoring modelu po wdrożeniu: porównanie predykcji z rzeczywistym churnem, kontrola dryfu danych, okresowe ponowne trenowanie.
- Zweryfikowanie wniosków na innym zbiorze danych lub na danych rzeczywistych.
