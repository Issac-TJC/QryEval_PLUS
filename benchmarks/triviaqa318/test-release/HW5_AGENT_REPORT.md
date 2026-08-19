# Agentic Retrieval comparison

The historical column is descriptive. Agent lift is attributed only against the contemporaneous fixed baseline.
The historical values cover the original 40 questions; they are not a five-question smoke-test score.

| System | EM | F1 | MRR | P@1 | P@5 | Time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fixed BM25 | 74.46 | 82.38 | 0.7924 | 0.7194 | 0.5338 | 31:05 |
| BM25 + RM3 PRF | 74.82 | 83.13 | 0.7606 | 0.6835 | 0.5432 | 62:25 |
| Always LLM Rewrite | 76.62 | 84.17 | 0.8518 | 0.8022 | 0.6554 | 50:23 |
| Adaptive LLM Rewrite | 75.90 | 84.18 | 0.8200 | 0.7482 | 0.5806 | 39:15 |

## Paired F1 differences

- BM25 + RM3 PRF: ΔF1=0.75, ΔEM=0.36, 95% F1 CI [-1.41, 2.99], randomization p=0.5076, Holm-adjusted p=0.5076 — positive point estimate, but evidence is insufficient.
- Always LLM Rewrite: ΔF1=1.79, ΔEM=2.16, 95% F1 CI [-0.67, 4.31], randomization p=0.1570, Holm-adjusted p=0.3140 — positive point estimate, but evidence is insufficient.
- Adaptive LLM Rewrite: ΔF1=1.79, ΔEM=1.44, 95% F1 CI [-0.30, 4.03], randomization p=0.1029, Holm-adjusted p=0.3087 — positive point estimate, but evidence is insufficient.

## Baseline difficulty strata

- relevant at rank 1 (n=200): BM25 + RM3 PRF ΔF1 +0.94; Always LLM Rewrite ΔF1 +0.82; Adaptive LLM Rewrite ΔF1 +1.25.
- relevant at ranks 2-10 (n=55): BM25 + RM3 PRF ΔF1 -1.15; Always LLM Rewrite ΔF1 +2.78; Adaptive LLM Rewrite ΔF1 +1.42.
- first relevant below rank 10 (n=14): BM25 + RM3 PRF ΔF1 -1.27; Always LLM Rewrite ΔF1 +9.64; Adaptive LLM Rewrite ΔF1 +9.32.
- not retrieved in judged depth (n=9): BM25 + RM3 PRF ΔF1 +11.11; Always LLM Rewrite ΔF1 +4.90; Adaptive LLM Rewrite ΔF1 +4.44.

## Agent operations

| System | Model calls | Retrievals | Tokens | p50 (s) | p95 (s) | p99 (s) | Cost/query | Cache hits | Rewrite rate | Error rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Fixed BM25 | 1.00 | 0.00 | 1242.6 | 0.99 | 1.23 | 1.54 | $0.000175 | 0 | 0.00% | 0.00% |
| BM25 + RM3 PRF | 1.00 | 0.00 | 1248.0 | 1.10 | 1.49 | 1.96 | $0.000151 | 39 | 0.00% | 0.00% |
| Always LLM Rewrite | 2.00 | 2.00 | 3054.9 | 10.35 | 16.07 | 19.36 | $0.000435 | 0 | 100.00% | 0.00% |
| Adaptive LLM Rewrite | 2.00 | 1.45 | 3072.4 | 7.89 | 13.86 | 16.17 | $0.000294 | 224 | 44.96% | 0.00% |

Tool-use totals:
- Fixed BM25: `{}`.
- BM25 + RM3 PRF: `{}`.
- Always LLM Rewrite: `{"rewrite_query": 278}`.
- Adaptive LLM Rewrite: `{"finish": 153, "rewrite_query": 125}`.

## Per-question gains and regressions

### BM25 + RM3 PRF

- Largest gains:
  - `qb_10019` (ΔF1 +100.00) The cathedral in which British city is known as ‘The Ship of the Fens’? — fixed=`Peterborough`, agent=`Ely.`
  - `qw_15678` (ΔF1 +100.00) The Council of Trent in the 16th century was held between believers of what religious faith? — fixed=`Catholic Church`, agent=`Catholicism.`
  - `jp_2516` (ΔF1 +100.00) What is the most popular ice cream flavor in America? — fixed=`The context does not state the most popular flavor.`, agent=`Vanilla.`
  - `sfq_14550` (ΔF1 +100.00) Who wrote the Mott The Hoople hit 'All The Young Dudes'? — fixed=`Ian Hunter.`, agent=`David Bowie.`
  - `odql_1002` (ΔF1 +100.00) Between 1959 and 1967 which city was the capital of Pakistan (Islamabad was being built)? — fixed=`Karachi`, agent=`Rawalpindi`
  - `sfq_8996` (ΔF1 +73.33) Actor Norman Painting died in November 2009, which part in a log running radio series did he make his own? — fixed=`Norman Painting played the part of Phil Archer in the long-running radio series The Archers.`, agent=`Phil Archer`
  - `jp_3026` (ΔF1 +50.00) What does an entomologist study? — fixed=`Entomologists study insects.`, agent=`Insects.`
  - `bb_9471` (ΔF1 +33.33) Originating in early central American culture, where on the body would a huarache be worn? — fixed=`On the feet.`, agent=`Foot.`
  - `qw_7650` (ΔF1 +33.33) "What profession had been followed by Yorick, a character in Shakespeare's ""Hamlet""?" — fixed=`Yorick was a court jester.`, agent=`Jester.`
  - `qw_15171` (ΔF1 +26.67) "Joss Whedon's 2002 US TV series ""Firefly"" won a 2003 Primetime Emmy Award for what?" — fixed=`Outstanding Special Class Program.`, agent=`Outstanding Special Visual Effects for a Miniseries, Movie or a Special.`
  - `qw_6502` (ΔF1 +23.08) The island of Taiwan is off the coast of where? — fixed=`The southeastern coast of the People's Republic of China.`, agent=`China`
  - `odql_9121` (ΔF1 +13.33) Who was the King of Libya who was overthrown by a military coupled by Colonel Qaddafi in 1969? — fixed=`King Idris.`, agent=`King Idris I.`
  - `sfq_7117` (ΔF1 +9.09) What was the name of the Edinburgh dog that watched over his owner's grave for 14 years? — fixed=`The provided text does not contain the name of the Edinburgh dog.`, agent=`The provided text does not contain information about an Edinburgh dog that watched over his owner's grave for 14 years.`
  - `dpql_277` (ΔF1 +4.44) Which title character was named Dolores Haze? — fixed=`Dolores Haze is the title character in the novel *Lolita*.`, agent=`Dolores Haze is the title character of Lolita.`
  - `sfq_24030` (ΔF1 +1.79) b Who was the tallest British Prime Minister of the 20th century? — fixed=`The provided text does not contain information about the heights of British Prime Ministers.`, agent=`The tallest British Prime Minister of the 20th century was Harold Macmillan.`
- Largest regressions:
  - `sfq_3816` (ΔF1 -100.00) After who is the Belfast City Airport named? — fixed=`George Best.`, agent=`The passage does not specify after whom Belfast City Airport is named.`
  - `qz_186` (ΔF1 -77.78) What is the most popular natural attraction in Alaska? — fixed=`Denali`, agent=`The most popular natural attraction in Alaska is Denali.`
  - `odql_2538` (ΔF1 -66.67) Which ball sport is played on a court 18 metres long by 9 metres wide? — fixed=`Sitting volleyball.`, agent=`Throwball.`
  - `odql_4001` (ΔF1 -63.64) Who did 'Eastenders' character 'Dot Cotton' marry in 2001? — fixed=`Jim Branning`, agent=`Dot Cotton married Jim Branning in 2002, not 2001.`
  - `qw_13108` (ΔF1 -60.00) What, according to Greek legend, was built at Cnossos, Crete, by Daedalus? — fixed=`The Labyrinth.`, agent=`A labyrinth to imprison the Minotaur.`
  - `qw_14136` (ΔF1 -55.56) The ship Andrea Gail features in which 1997 book and 2000 film? — fixed=`The Perfect Storm`, agent=`The Perfect Storm (book) and The Perfect Storm (film).`
  - `dpql_3499` (ΔF1 -42.86) Which is the only team to have won football’s Premier League once and once only? — fixed=`Blackburn Rovers.`, agent=`Leicester City`
  - `qg_3242` (ΔF1 -33.33) Where is the human body do you find the hammer, anvil, and stirrup? — fixed=`The ear.`, agent=`In the ear.`
  - `tb_1852` (ΔF1 -33.33) Which 1989 music video of Madonna attracted criticism for showing images like her making love to Saint Martin de Porres, use of Catholic iconography including a scene where she develops stigmata as well as cross burning? — fixed=`Like a Prayer`, agent=`The "Like a Prayer" music video.`
  - `sfq_2723` (ΔF1 -14.29) Which shipyard built the 'Titanic'? — fixed=`Harland and Wolff shipyard.`, agent=`Harland & Wolff shipyard.`
  - `sfq_15005` (ΔF1 -9.52) Who was on the English throne when Jack Cade led the Men of Kent in a short lived revolt? — fixed=`Henry VI of England.`, agent=`King Henry VI of England.`
  - `bb_291` (ΔF1 -3.81) In climatology, pluvial relates to what? — fixed=`Pluvial relates to periods of increased rainfall or moisture, often associated with ancient lakes.`, agent=`Pluvial relates to periods of increased moisture and higher lake levels, often associated with past climate conditions.`
- Recorded failures: 0.
### Always LLM Rewrite

- Largest gains:
  - `qb_10019` (ΔF1 +100.00) The cathedral in which British city is known as ‘The Ship of the Fens’? — fixed=`Peterborough`, agent=`Ely`
  - `sfq_7117` (ΔF1 +100.00) What was the name of the Edinburgh dog that watched over his owner's grave for 14 years? — fixed=`The provided text does not contain the name of the Edinburgh dog.`, agent=`Greyfriars Bobby`
  - `sfq_11011` (ΔF1 +100.00) Which modern musical instrument was developed from the shawm? — fixed=`The modern tárogató was developed from the shawm.`, agent=`The oboe.`
  - `sfq_14550` (ΔF1 +100.00) Who wrote the Mott The Hoople hit 'All The Young Dudes'? — fixed=`Ian Hunter.`, agent=`David Bowie`
  - `odql_1002` (ΔF1 +100.00) Between 1959 and 1967 which city was the capital of Pakistan (Islamabad was being built)? — fixed=`Karachi`, agent=`Rawalpindi`
  - `bb_291` (ΔF1 +86.67) In climatology, pluvial relates to what? — fixed=`Pluvial relates to periods of increased rainfall or moisture, often associated with ancient lakes.`, agent=`Rain.`
  - `dpql_277` (ΔF1 +60.00) Which title character was named Dolores Haze? — fixed=`Dolores Haze is the title character in the novel *Lolita*.`, agent=`Lolita.`
  - `jp_3026` (ΔF1 +50.00) What does an entomologist study? — fixed=`Entomologists study insects.`, agent=`Insects.`
  - `sfq_2754` (ΔF1 +50.00) What was the original use of the building which now houses the Tate Modern Art Gallery in London? — fixed=`The building was originally a power station.`, agent=`Bankside Power Station.`
  - `odql_5979` (ΔF1 +50.00) Which company manufactures the Playstation video games console? — fixed=`Sony Interactive Entertainment, a division of Sony.`, agent=`Sony`
  - `qw_15171` (ΔF1 +40.00) "Joss Whedon's 2002 US TV series ""Firefly"" won a 2003 Primetime Emmy Award for what?" — fixed=`Outstanding Special Class Program.`, agent=`Outstanding Special Visual Effects.`
  - `odql_10620` (ΔF1 +40.00) Which acid is contained in Grapes? — fixed=`Tartaric acid and malic acid.`, agent=`Tartaric acid.`
  - `qw_10268` (ΔF1 +37.14) "In Charles Dickens' ""Great Expectations"", who or what was Abel Magwitch?" — fixed=`Abel Magwitch was a convict in Charles Dickens' "Great Expectations.`, agent=`Abel Magwitch was an escaped convict.`
  - `bb_9471` (ΔF1 +33.33) Originating in early central American culture, where on the body would a huarache be worn? — fixed=`On the feet.`, agent=`Foot.`
  - `qw_7650` (ΔF1 +33.33) "What profession had been followed by Yorick, a character in Shakespeare's ""Hamlet""?" — fixed=`Yorick was a court jester.`, agent=`Jester.`
- Largest regressions:
  - `qg_1459` (ΔF1 -100.00) Marking the separation between the Atlantic and Pacific Oceans, what name is given to the southernmost piece of South America? — fixed=`Cape Horn`, agent=`Tierra del Fuego`
  - `sfq_22920` (ΔF1 -100.00) Who was the choreographer of the dance troupe Hot Gossip? — fixed=`Arlene Phillips`, agent=`Flick Colby.`
  - `sfq_4319` (ΔF1 -80.00) Who wounded Achilles in the heel, leading to his death? — fixed=`Paris`, agent=`Paris wounded Achilles in the heel, leading to his death.`
  - `qz_186` (ΔF1 -66.67) What is the most popular natural attraction in Alaska? — fixed=`Denali`, agent=`Denali National Park and Preserve.`
  - `odql_4001` (ΔF1 -63.64) Who did 'Eastenders' character 'Dot Cotton' marry in 2001? — fixed=`Jim Branning`, agent=`Dot Cotton married Jim Branning in 2002, not 2001.`
  - `qw_13108` (ΔF1 -60.00) What, according to Greek legend, was built at Cnossos, Crete, by Daedalus? — fixed=`The Labyrinth.`, agent=`A labyrinth to imprison the Minotaur.`
  - `dpql_3499` (ΔF1 -42.86) Which is the only team to have won football’s Premier League once and once only? — fixed=`Blackburn Rovers.`, agent=`Leicester City`
  - `tc_3130` (ΔF1 -30.00) What is the Milky Way? — fixed=`The Milky Way is the galaxy that contains our Solar System.`, agent=`The Milky Way is a barred spiral galaxy that contains our Solar System, Earth, and hundreds of billions of stars.`
  - `qw_11570` (ΔF1 -25.00) What kind of bird is a guillemot? — fixed=`A guillemot is a seabird.`, agent=`A guillemot is a seabird in the auk family, Alcidae.`
  - `qz_3358` (ΔF1 -23.08) Who is the patron saint of lost causes? — fixed=`Saint Jude.`, agent=`Saint Jude is the patron saint of lost causes.`
  - `sfq_2723` (ΔF1 -14.29) Which shipyard built the 'Titanic'? — fixed=`Harland and Wolff shipyard.`, agent=`Harland & Wolff shipyard.`
  - `dpql_6346` (ΔF1 -13.89) Who was Wolfe’s opposing French general who died with him on the Plains of Abraham? — fixed=`Louis-Joseph de Montcalm`, agent=`The Marquis de Montcalm.`
  - `sfq_24030` (ΔF1 -12.50) b Who was the tallest British Prime Minister of the 20th century? — fixed=`The provided text does not contain information about the heights of British Prime Ministers.`, agent=`The provided text does not contain information about British Prime Ministers.`
  - `qw_14447` (ΔF1 -2.42) "In which "" Star Wars"" episode does Anakin Skywalker transform into the black-suited Darth Vader?" — fixed=`Star Wars: Episode III – Revenge of the Sith`, agent=`Episode III – Revenge of the Sith`
- Recorded failures: 0.
### Adaptive LLM Rewrite

- Largest gains:
  - `qb_10019` (ΔF1 +100.00) The cathedral in which British city is known as ‘The Ship of the Fens’? — fixed=`Peterborough`, agent=`Ely`
  - `jp_2516` (ΔF1 +100.00) What is the most popular ice cream flavor in America? — fixed=`The context does not state the most popular flavor.`, agent=`Vanilla.`
  - `sfq_7117` (ΔF1 +100.00) What was the name of the Edinburgh dog that watched over his owner's grave for 14 years? — fixed=`The provided text does not contain the name of the Edinburgh dog.`, agent=`Greyfriars Bobby`
  - `sfq_14550` (ΔF1 +100.00) Who wrote the Mott The Hoople hit 'All The Young Dudes'? — fixed=`Ian Hunter.`, agent=`David Bowie`
  - `odql_1002` (ΔF1 +100.00) Between 1959 and 1967 which city was the capital of Pakistan (Islamabad was being built)? — fixed=`Karachi`, agent=`Rawalpindi`
  - `bb_291` (ΔF1 +86.67) In climatology, pluvial relates to what? — fixed=`Pluvial relates to periods of increased rainfall or moisture, often associated with ancient lakes.`, agent=`Rain.`
  - `dpql_277` (ΔF1 +60.00) Which title character was named Dolores Haze? — fixed=`Dolores Haze is the title character in the novel *Lolita*.`, agent=`Lolita`
  - `sfq_2754` (ΔF1 +50.00) What was the original use of the building which now houses the Tate Modern Art Gallery in London? — fixed=`The building was originally a power station.`, agent=`Bankside Power Station.`
  - `qw_15171` (ΔF1 +40.00) "Joss Whedon's 2002 US TV series ""Firefly"" won a 2003 Primetime Emmy Award for what?" — fixed=`Outstanding Special Class Program.`, agent=`Outstanding Special Visual Effects.`
  - `qw_10268` (ΔF1 +37.14) "In Charles Dickens' ""Great Expectations"", who or what was Abel Magwitch?" — fixed=`Abel Magwitch was a convict in Charles Dickens' "Great Expectations.`, agent=`Abel Magwitch was an escaped convict.`
  - `qw_7650` (ΔF1 +33.33) "What profession had been followed by Yorick, a character in Shakespeare's ""Hamlet""?" — fixed=`Yorick was a court jester.`, agent=`Jester`
  - `wh_4195` (ΔF1 +16.67) What type of creature is a margay ? — fixed=`A margay is a type of wild cat.`, agent=`A margay is a wild cat.`
- Largest regressions:
  - `sfq_22920` (ΔF1 -100.00) Who was the choreographer of the dance troupe Hot Gossip? — fixed=`Arlene Phillips`, agent=`Flick Colby.`
  - `sfq_4319` (ΔF1 -80.00) Who wounded Achilles in the heel, leading to his death? — fixed=`Paris`, agent=`Paris wounded Achilles in the heel, leading to his death.`
  - `qz_186` (ΔF1 -66.67) What is the most popular natural attraction in Alaska? — fixed=`Denali`, agent=`Denali National Park and Preserve.`
  - `dpql_3499` (ΔF1 -42.86) Which is the only team to have won football’s Premier League once and once only? — fixed=`Blackburn Rovers.`, agent=`Leicester City`
  - `qz_3358` (ΔF1 -23.08) Who is the patron saint of lost causes? — fixed=`Saint Jude.`, agent=`Saint Jude is the patron saint of lost causes.`
  - `sfq_24030` (ΔF1 -12.50) b Who was the tallest British Prime Minister of the 20th century? — fixed=`The provided text does not contain information about the heights of British Prime Ministers.`, agent=`The provided text does not contain information about British Prime Ministers.`
- Recorded failures: 0.
