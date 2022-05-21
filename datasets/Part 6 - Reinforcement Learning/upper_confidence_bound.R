
# Upper Confidence Bound

setwd("~/repos/machinelearning-az/datasets/Part 6 - Reinforcement Learning")

# Importar el dataset
dataset = read.csv('data/Ads_CTR_Optimisation.csv')

# Implementar UCB
d = 10
N = 10000
number_of_selections = integer(d)
sums_of_rewards = integer(d)
ads_selected = integer(0)
total_reward = 0
for(n in 1:N){
  max_upper_bound = 0
  ad = 0
  for(i in 1:d){
    if(number_of_selections[i]>0){
      average_reward = sums_of_rewards[i]/number_of_selections[i]
      delta_i = sqrt(3/2*log(n)/number_of_selections[i])
      upper_bound = average_reward + delta_i
    }
    else{
      upper_bound = 1e400
    }
    if(upper_bound > max_upper_bound){
      max_upper_bound = upper_bound
      ad = i
    }
  }
  ads_selected = append(ads_selected, ad)
  number_of_selections[ad] = number_of_selections[ad] + 1
  reward = dataset[n, ad]
  sums_of_rewards[ad] = sums_of_rewards[ad] + reward
  total_reward = total_reward + reward
}

# Visualizar los resultados - Histograma
hist(ads_selected,
     col = 'lightblue',
     main = 'Histograma de anuncios',
     xlab = 'ID del Anuncio',
     ylab = 'Frecuencia absoluta del anuncio')
