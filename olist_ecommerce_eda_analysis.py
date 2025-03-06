#%%
# Repliquei um projeto do Kaggle no VS Code para praticar e consolidar minhas habilidades em análise e visualização de dados. 
# Creditos: https://www.kaggle.com/code/phiphatjan/sql-and-python-in-depth-analysis#4.-Create-SQLite-Engine-&-Populate-Tables

# Codigo em andamento !

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import seaborn as sns
from sqlalchemy import create_engine

#%%

# Carregando os arquivos separadamente:

customers = pd.read_csv("olist_customers_dataset.csv")
geolocation = pd.read_csv("olist_geolocation_dataset.csv")
order_items = pd.read_csv("olist_order_items_dataset.csv")
order_payments = pd.read_csv("olist_order_payments_dataset.csv")
order_reviews = pd.read_csv("olist_order_reviews_dataset.csv")
orders = pd.read_csv("olist_orders_dataset.csv")
products = pd.read_csv("olist_products_dataset.csv")
sellers = pd.read_csv("olist_sellers_dataset.csv")
category = pd.read_csv("product_category_name_translation.csv")

#%%

# Limpando os dados em brancos e nulos de cada dataframe:

table_name = [customers, geolocation, order_items,order_payments, order_reviews, orders, products, sellers, category]

def check_null(filename):
    shape = filename.shape
    null_counts = filename.isna().sum()
    return shape, null_counts

for i in table_name:
    print(check_null(i))


# Populando as reviews que estao sem comentarios / titulo:

order_reviews["review_comment_title"] = order_reviews["review_comment_title"].fillna("No title")
order_reviews["review_comment_message"] = order_reviews["review_comment_message"].fillna("No comment")

order_reviews.isna().sum()
order_reviews.shape

#%%


# Removendo linhas que estao nulas do arquivo de orders :

initial_row_orders = orders.shape[0]
orders.dropna(subset=["order_approved_at", "order_delivered_carrier_date", "order_delivered_customer_date"], inplace=True)
final_rows_orders = orders.shape[0]
print(orders.isna().sum())

# Garantindo que o numero de linhas no DF com valores nulos diminua menos de 5%
check_percent_orders = ((initial_row_orders - final_rows_orders) / final_rows_orders) * 100
print("Porcentagem de linhas nulas iniciais e finais:", check_percent_orders)


#%%


# Removendo as linhas nulas do arquivo de products :


# contando quantas linhas existem na tabela orders antes de remover os valores nulos:
initial_row_prod = products.shape[0] # o shape[0] retorna o número total de linhas do DataFrame

# Removendo as linhas nulas das colunas que tem valores nulos :
products.dropna(subset=["product_category_name", "product_name_lenght", "product_description_lenght", "product_photos_qty", \
                        "product_weight_g", "product_length_cm", "product_height_cm", "product_width_cm"],
                         inplace=True)

# quanto numero total de linhas que restaram na tabela orders após a remoção dos valores nulos: 
final_rows_prod = products.shape[0] 

# quantos valores nulos existem em cada coluna do DataFrame após a remoção: (ideal = zero)
print("Qtd de nulos:", products.isna().sum()) 

# Garantindo que o numero de linhas com valores nulos removidas do DF foi menos de 5% (para não perder muitos dados):
check_percent_prod = ((initial_row_prod - final_rows_prod) / final_rows_prod) * 100
print("Porcentagem de linhas nulas removidas:", check_percent_prod)



#*************POR QUE É IMPORTANTE TER O CONTROLE DE PERDAS DE DADOS ?????

# Verificar a porcentagem de linhas removidas ajuda a garantir que você não está perdendo muitos dados. 
# Se mais de 5% dos dados forem removidos, pode ser necessário repensar a estratégia 
# (por exemplo, preencher os valores nulos em vez de removê-los).


#%%


                        # CRIANDO UMA ENGINE EM SQLITE PARA MANIPULAR EM SQL NO PYTHON :

engine = create_engine('sqlite://', echo=False)

        # O que faz?

# A função create_engine é usada para criar uma conexão com um banco de dados. No caso, estamos criando um banco de dados SQLite em memória (não salvo em disco).
# O argumento 'sqlite://' especifica que estamos usando o SQLite e que o banco de dados será temporário (em memória).
# O argumento echo=False desativa a exibição de logs das operações SQL executadas (útil para evitar poluição visual no console).


customers.to_sql('customers', con=engine, index=False, if_exists='replace')
geolocation.to_sql('geolocation', con=engine, index=False, if_exists='replace')
order_items.to_sql('order_items', con=engine, index=False, if_exists='replace')
order_payments.to_sql('order_payments', con=engine, index=False, if_exists='replace')
order_reviews.to_sql('order_reviews', con=engine, index=False, if_exists='replace')
orders.to_sql('orders', con=engine, index=False, if_exists='replace')
products.to_sql('products', con=engine, index=False, if_exists='replace')
sellers.to_sql('sellers', con=engine, index=False, if_exists='replace')
category.to_sql('category', con=engine, index=False, if_exists='replace')


        # O que faz?

# nome_tabela.to_sql : salva o DataFrame customers em uma tabela chamada 'customers' no banco de dados SQLite que criamos.

# 'customers': Nome da tabela que será criada no banco de dados.
# con=engine: Especifica a conexão com o banco de dados (no caso, o engine que criamos).
# index=False: Evita que o índice do DataFrame seja salvo como uma coluna na tabela SQL.
# if_exists='replace': Se a tabela já existir, ela será substituída. Outras opções são 'fail' (lança um erro) ou 'append' (adiciona os dados ao final da tabela existente).

        # A RESPOSTA DO CODIGO : 71 --> significa que 71 linhas de todas as tabelas convertidas em sql



#%%

                             # ANALISE DE SERIES TEMPORAIS DOS DADOS (JA LIMPOS)

                            # VENDAS E QTD DE ORDERS MENSAL:

# Preparando os dados:
    #1: Merge data = Juntar as tabelas de orders, order_items e products tables baseado em order_id(orders e order_items) e product_id(products).
    #2: Filler data = Incluir apenas orders com order_status 'delivered'.
    #3: Aggregate data = agrupar a data por mes e ano (year_month)
    #4: Calculos = 
                # total_price = soma do preço do produto e do frete
                # orders_per_mth = conta a qtd de orders para cada mes
    #5: Sort data = ordernar os dados por year_month.

merge_order_orderitems = """
                            WITH merge_data AS (
                            SELECT *
                            FROM orders o 
                            INNER JOIN order_items oi ON o.order_id = oi.order_id
                            INNER JOIN products p ON oi.product_id = p.product_id
                            )
                            
                            SELECT
                                strftime('%Y', order_purchase_timestamp) || '-' || strftime('%m', order_purchase_timestamp) as year_month,
                                SUM(price + freight_value) as total_price,
                                COUNT(order_status) as orders_per_mth
                            FROM merge_data
                            WHERE order_status = 'delivered'
                            GROUP BY year_month
                            ORDER BY year_month
                            LIMIT 100;
                            
                            """
                        # EXPLICAÇAO DO CODIGO DE SQL ACIMA :

                        # strftime('%Y', order_purchase_timestamp): Extrai o ano da coluna order_purchase_timestamp.
                        # strftime('%m', order_purchase_timestamp): Extrai o mês da coluna order_purchase_timestamp.
                        # || : Combina o ano e o mês no formato YYYY-MM (por exemplo, 2023-10).
                        # AS year_month: Dá um nome à coluna calculada.


merge_order_orderitems = pd.read_sql_query(merge_order_orderitems, engine)
print(merge_order_orderitems) 

#%%

                            # VISUALIZACAO GRAFICA DA TABELA ACIMA:

    #1: Formatando a area plotada = Duas figuras (com duas analises distintas) na esqueda e texto na direita.
    #2: Texto = com o valor total de vendas, total de orders e o periodo de datas analisado.  
    #3: Grafico da Receita = grafico de linhas que mostra a receita ao longo dos meses.
    #4: Grafico da Qtd de orders = grafico de linhas que mostra a qtd de orders feitas ao longo dos meses.


# Transformando o type da coluna year_month em data para manipulacao:
merge_order_orderitems['year_month'] = pd.to_datetime(merge_order_orderitems['year_month'], errors='coerce')


# Formatando valores muito grandes para que apareçam de forma abreviada (K, M):
def formato_num_y(valor,_): # Estrutura da funçao criada onde o '_' siginifica que um segundo valor nao sera necessario -- escreve-se None.
    if valor >= 1e6: # Se o valor for maior ou igual a 1 milhão 
        return f'{valor/1e6:.1f}M' # Divide o valor por 1 milhao / Formata o resultado com 1 casa decimal / adiciona o sufixo M (ex: 1500000 -> 1500000/1000000 = 1.5)
    elif valor >= 1e3: # Se o valor for maior ou igual a 1 mil
        return f'{valor/1e3:.1f}K' # Divide o valor por 1000(mil) / Formata o resultado com 1 casa decimal / adiciona o sufixo K (ex: 1500 -> 1500/1000 = 1.5)
    else:
        return f'{valor:.0f}' # Sem abreviaçoes, sufixos.


# Definindo o tamanho das fontes: 
fonte_peq = 14
fonte_med = 18
fonte_grande = 25


# Definindo a area de plotagem :
plt.figure(figsize=(24,18))
gs = gridspec.GridSpec(2, 2, width_ratios=[1,2], height_ratios=[1,1])


# Formando o texto que sera exibido a direita :
ax0 = plt.subplot(gs[:,0])
ax0.axis('off')
texto = (
        f"Receita de vendas: ${merge_order_orderitems['total_price'].sum():,.2f}\n" # Formata o valor com 2 casas decimais / e o '\n' = pula uma linha.
        f"Qtd de orders: {merge_order_orderitems['orders_per_mth'].sum():,}\n"
        f"Periodo analisado: Set/2016 a Ago/2019"
        )
#f"Periodo analisado: {merge_order_orderitems['year_month'][0]} a {merge_order_orderitems['year_month'][-1]}"
ax0.text(0.6, 0.5, texto, ha='center', va='center', fontsize=fonte_grande, wrap=True)

        # EXPLICACAO DO CODIGO DA FORMATACAO DO TEXTO ACIMA :
                
                #1: plt.subplot(gs[:, 0]) = Cria um subplot (ax0) que ocupa toda a primeira coluna da grade.
                    # Isso significa que, se a grade tiver 2 linhas e 2 colunas, ax0 ocupará a primeira coluna inteira (as duas células da primeira coluna).
                #2: gs[:, 0] =
                    # Seleciona todas as linhas (:) da primeira coluna (0) da grade criada anteriormente com GridSpec. 
                #3: ax0.axis('off') = 
                    # Desativa a exibição dos eixos (linhas de borda, rótulos, etc.) no subplot ax0.
                #4: merge_order_orderitems['year_month'][0] & Usa o primeiro valor ([0]) e merge_order_orderitems['year_month'][-1] o último valor ([-1]) da coluna year_month para indicar o intervalo de tempo analisado.
                
                #5: ax0.text() = Adiciona o texto ao subplot ax0.
                    # 0.6, 0.5: Coordenadas (x, y) onde o texto será posicionado. Esses valores são normalizados, ou seja, (0, 0) é o canto inferior esquerdo e (1, 1) é o canto superior direito.
                    # texto = a variavel texto que sera exibida.
                    # ha='center' = Alinha o texto horizontalmente ao centro (ha significa "horizontal alignment").
                    # va='center'= Alinha o texto verticalmente ao centro (va significa "vertical alignment").
                    # fontsize=fonte_grande = Define o tamanho da fonte. Aqui, fonte_grande é uma variável que deve ter sido definida anteriormente no código.
                    # wrap=True = Permite que o texto seja quebrado em várias linhas caso ultrapasse a largura do subplot.


# Criando o 1o grafico :
ax1 = plt.subplot(gs[0,1]) # Primeira linha, segunda coluna.
ax1.plot(merge_order_orderitems['year_month'], merge_order_orderitems['total_price'], marker='o', linewidth=2.5, markersize=8, color='royalblue')
ax1.set_ylabel("Receita", fontsize=fonte_med, fontweight='bold')
ax1.set_xlabel("Mes-Ano", fontsize=fonte_med, fontweight='bold')
ax1.set_title("Receita Mensal", fontsize=fonte_grande, fontweight='bold')

    # Formatando o eixo Y de acordo com a funcao personalizada para abreviar grandes valores :
ax1.yaxis.set_major_formatter(ticker.FuncFormatter(formato_num_y))
    # Configura o eixo X para exibir rótulos a cada 1 mês (itervail = 1):
ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    # Formatando o tamanho e a rotacao dos meses, ou seja, o eixo X :
ax1.tick_params(axis='x', labelsize=fonte_peq, rotation=45)

    # Formatando o valores de rotulo do grafico a cada mes :
for i, txt in enumerate(merge_order_orderitems['total_price']):
    ax1.annotate(
        f'{txt/1e6:.2f}M', # Texto a ser exibido
        (merge_order_orderitems['year_month'].iloc[i], txt), # Posição do ponto
        textcoords="offset points", # Define que o deslocamento do texto (xytext) será em pontos (unidade de medida relativa).
        xytext=(0,10), # significa que o texto será deslocado 0 pontos horizontalmente e 10 pontos verticalmente (acima do ponto).
        ha='center', # alinhamento horizontal do texto como centralizado.
        fontsize=fonte_peq # Tamanho da fonte
    )


# Criando o 2o grafico:
ax2 = plt.subplot(gs[1,1]) # segunda linha, segunda coluna
ax2.plot(merge_order_orderitems['year_month'], merge_order_orderitems['orders_per_mth'], marker='o', linewidth=2.5, markersize=8, color='royalblue')
ax2.set_xlabel("Mes-Ano", fontsize=fonte_med, fontweight='bold')
ax2.set_ylabel("Qtd de orders", fontsize=fonte_med, fontweight='bold')
ax2.set_title("Qtd de orders Mensal", fontsize=fonte_grande, fontweight='bold')

    # Formatabdo o eixo Y de acordo com a funcao personalizada pra abreviar grandes valores:
ax2.yaxis.set_major_formatter(ticker.FuncFormatter(formato_num_y))
    # Formatando o eixo X para exibir rotulos a cada 1 mes sem falhas:
ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    # Formatando o tamanho e a rotacao dos meses no eixo X:
ax2.tick_params(axis='x', labelsize=fonte_peq, rotation=45)

    # Formatando o valores de rotulo dentro do grafico a cada mes:
for i, txt in enumerate(merge_order_orderitems['orders_per_mth']):
    ax2.annotate(
        txt,
        (merge_order_orderitems['year_month'].iloc[i], txt),
        textcoords="offset points",
        xytext=(0,10),
        ha='center',
        fontsize=fonte_peq
    )


# Graficos prontos pra exibir:
plt.tight_layout() # Evitar ocorrer sobreposicao e corte dos graficos e do texto
plt.show()




#%%

                        # ANALISE DA ENTREGA REAL E ESTIMADA POR ESTADO & CIDADE

# Preparando os dados:
    #1 Merge tables = pegar as tabelas usando SQL de orders, order_items, order_reviews, customers, and geolocation a partir da chave primaria comum
# order_id customer_zip_code_prefix com INNER JOIN.
    #2 Sem filtro de dados = incluir todos os dados (sem necessidade de where)
    #3 Agrupar dados para analise da discrepancia (data real - estimada) : 
            # Por estado = agrupar dado pelo estado do customer e pela media da qtd de pedidos entregues
            # Por cidade = agrupar dado pelo cidade do customer e pela media da qtd de pedidos entregues
            # Incluir apenas cidades com discrepancias positivas (chefou mais rapido que o estimado/previsto)
    #4 Metricas = media da discrepancia da entrega e a qtd de pedidos por estado e por cidade
    #5 Ordernar os dados :
            # Decrescente = discrepancia da media de entrega (do maior pro menor)
            # Limitar resultados para apenas as 10 cidades e estados com maior discrepancia.


#1 Merge tables usando SQL :
delivery_discrep_state = """
                        SELECT
                        c.customer_state,
                        AVG( julianday(strftime('%d', order_delivered_customer_date)) - julianday(strftime('%d', order_estimated_delivery_date)) ) as avg_delivery_discrep,
                        count(o.order_id) as order_count

                        FROM orders o 
                        INNER JOIN order_items i ON o.order_id = i.order_id
                        INNER JOIN order_reviews r ON i.order_id = r.order_id
                        INNER JOIN customers c ON o.customer_id = c.customer_id
                        INNER JOIN geolocation g ON g.geolocation_zip_code_prefix = c.customer_zip_code_prefix

                        GROUP BY c.customer_state
                        ORDER BY avg_delivery_discrep DESC
                        LIMIT 10;
                        """
                        # DETALHE DO CODIGO de SQL:
                        # o uso do julianday é necessario pq o sqlite nao permite que operacoes matematicas sejam feitas diretamente pelo strftime


delivery_state_perf = pd.read_sql_query(delivery_discrep_state, engine)
delivery_state_perf.drop_duplicates(inplace=True)
delivery_state_perf.dropna(inplace=True)
delivery_state_perf['avg_delivery_discrep'] = delivery_state_perf['avg_delivery_discrep'].apply(lambda x: f'{x:.2f}')

print(delivery_state_perf)

#%%

            # O MESMO POR CIDADE :

del_discip_city = """
                    SELECT
                    UPPER((c.customer_city || " ("||c.customer_state||")") ) as customer_city,
                    AVG( julianday(strftime('%d', order_delivered_customer_date)) - julianday(strftime('%d', order_estimated_delivery_date)) ) as avg_del_city,
                    COUNT(o.order_id) as qtd_orders

                    FROM orders o
                    INNER JOIN order_items i on o.order_id = i.order_id
                    INNER JOIN order_reviews r on o.order_id = r.order_id
                    INNER JOIN customers c ON o.customer_id = c.customer_id
                    INNER JOIN geolocation g ON g.geolocation_zip_code_prefix = c.customer_zip_code_prefix

                    GROUP BY c.customer_city
                    ORDER BY avg_del_city DESC
                    LIMIT 10;
                    """
del_city_perf = pd.read_sql_query(del_discip_city, engine)
del_city_perf.drop_duplicates(inplace=True)
del_city_perf.dropna(inplace=True)

print(del_city_perf)

#%%

                        # VISUALIZACAO GRAFICA DOS DADOS ACIMA:
            #1: Formatando a area plotada = Duas figuras (com duas analises distintas) no centri e texto acima.
            #2: Texto = 
                        # top 3 estados com a maior media de discrepancia de entrega. 
                        # top 3 cidades com a maior media de discrepancia da entrega.
            #3: Grafico dos estados = grafico de barras mostrando a discrepancia da entrega por estado de forma decrescente (maior >> menor)
            #4: Grafico das cidades =  grafico de barras mostrando a discrepancia da entrega por cidade de forma decrescente (maior >> menor)

#1: Relembrando o padrao definido anteriormente do tamanho das fontes: 
#fonte_peq = 14
#fonte_med = 18
#fonte_grande = 25


#1: Definindo a area de plotagem :
plt.figure(figsize=(24,18))
gs = gridspec.GridSpec(2, 2, width_ratios=[1,1], height_ratios=[0.2,1])

#2: Fromatando a area do texto :
ax4 = plt.subplot(gs[0,:]) # Primeira linha das duas colunas da figura
ax4.axis('off') # remove as linhas de grade


    #2.1: Transformando o tipo coluna de media em uma coluna numerica :
delivery_state_perf['avg_delivery_discrep'] = pd.to_numeric(delivery_state_perf['avg_delivery_discrep'], errors='coerce')
del_city_perf['avg_del_city'] = pd.to_numeric(del_city_perf['avg_del_city'], errors='coerce')
        
    #2.2: Calculando o top3:
top3_states = ( delivery_state_perf
               .nlargest(3, 'avg_delivery_discrep') # top3 maiores valores da coluna avg_delivery_discrep
               ['customer_state'] # mostra no resultado os estados com as top3 maiores valores de media
               .tolist() # cria uma lista [] com esses estados
              )

top3_cities = (del_city_perf
               .nlargest(3, 'avg_del_city')
               ['customer_city']
               .tolist()
               )


#2: Montando como o texto vai aparecer no topo da figura (em partes devido a diff de formatacao de cada linha - formata no prox codigo) :
part1 = "Os TOP 3 estados com a maior discrepancia media de atraso da entrega são:"
part2 = f"{', '.join(top3_states)}\n"
part3 = "As TOP 3 cidades com a maior discrepancia media de atraso da entrega são:"
part4 = f"{', '.join(top3_cities)}"

#2: Formatando os estilos de cada parte do texto :
ax4.text(0.5, 0.8, part1, ha='center',va='center', fontsize=fonte_med, wrap=True)
ax4.text(0.5, 0.55, part2, ha='center', va='center', fontsize=fonte_grande, weight='bold', wrap=True)
ax4.text(0.5, 0.45, part3, ha='center', va='center', fontsize=fonte_med, wrap=True)
ax4.text(0.5, 0.25, part4, ha='center', va='center', fontsize=fonte_grande, weight='bold', wrap=True)



#3: Grafico de barras dos TOP10 ESTADOS com maior discrepancia media :
ax5= plt.subplot(gs[1,0]) # gs[1,0] = 2a linha da 1a coluna
sns.barplot(x='avg_delivery_discrep', y='customer_state',data=delivery_state_perf, hue='customer_state', palette='YlGnBu', legend=False) # o hue =  defini grupos de cores no gráfico
ax5.set_title("Discrepancia Media por Estado", fontsize=fonte_med)
ax5.set_xlabel("Discrepancia Media (dias)", fontsize=fonte_peq)
ax5.set_ylabel("Estados", fontsize=fonte_peq)

        # Formatando o tamanho da legenda do eixo Y:
ax5.tick_params(axis='y', labelsize=fonte_peq)

        # Definindo os limites mínimos e máximos do eixo X (com o set_xlim() ) :
ax5.set_xlim(delivery_state_perf['avg_delivery_discrep'].min() - 0.2, # o menor valor do top10 - 0.2
             delivery_state_perf['avg_delivery_discrep'].max() + 0.2) # o maior valor do top10 + 0.2

        # Adicionando o valor ao lado de cada barra no grafico :
for bar, discrepancy in zip(ax5.patches, 
                            delivery_state_perf['avg_delivery_discrep']):
    width = bar.get_width() # calcula o comprimento da barra (a parte horizontal da barra)
    x_label_pos = width + 0.1 # adiciona mais um espaço pro valor nao sobrepor a barra
    ax5.text(x_label_pos, bar.get_y() + bar.get_height()/2, # agora calcula a parte vertical da barra pro valor ficar no meio, centralizado
             f'{discrepancy:.1f} dias', # o valor mais a unidade
             ha='center', # alinhamento horizando do texto
             va='center', # alinhamento vertical do texto
             fontsize=fonte_peq)



#4: Grafico de barras das TOP10 CIDADES com maior discrepancia media :
ax6= plt.subplot(gs[1,1]) # 2a linha da 2a coluna
sns.barplot(x='avg_del_city', y='customer_city', data=del_city_perf, hue='customer_city', palette='YlGnBu', legend=False)
ax6.set_title("Discrepancia Media por Cidade", fontsize=fonte_med)
ax6.set_xlabel("Discrepancia Media (dias)", fontsize=fonte_peq)
ax6.set_ylabel("Cidades", fontsize=fonte_peq)

        # Formatando o tamanho da legenda do eixo Y:
ax6.tick_params(axis='y', labelsize=fonte_peq)

        # Definindo os limites mínimos e máximos do eixo X (com o set_xlim() ) :
ax6.set_xlim(del_city_perf['avg_del_city'].min() - 15,
             del_city_perf['avg_del_city'].max() + 15 )

        # Adicionando o valor ao lado de cada barra no grafico :
for bar, discrepancy in zip(ax6.patches,
                            del_city_perf['avg_del_city']):
    width = bar.get_width()
    x_label_pos = width + 3
    ax6.text(x_label_pos, bar.get_y() + bar.get_height()/2,
             f'{discrepancy:.1f} dias',
             ha='center',
             va='center',
             fontsize=fonte_peq)

plt.tight_layout()
plt.show()


#%%

                                # COMPARACAO DE ENTREGAS ANTES E DEPOIS DO PRAZO
            # Usando o SQL :
            #1: Merge table = juntando as tabelas orders e order_delivery para saber as orders que foram entregues antes e depois do prazo
            #2: Metricas = 
                        #2.1: calcular as porcentagem de entregas antes e depois do prazo
                        #2.2: a qtd de orders entregues antes e depois do prazo
#1:
early_late_del = """
                WITH order_del AS (
                SELECT
                    order_id,
                    strftime('%d', order_delivered_customer_date) as delivered_date,
                    CASE 
                        WHEN order_estimated_delivery_date < order_delivered_customer_date --  data estimada pela loja MENOR QUE data que chegou no destinatario
                            THEN 'late_delivery' 
                        ELSE 'early_delivery' 
                    END AS delivery_status

                FROM orders
                    ),

                delivery_summary AS (
                SELECT
                    delivered_date,
                    SUM( CASE WHEN delivery_status = 'early_delivery' THEN 1 ELSE 0 END) as early,
                    SUM( CASE WHEN delivery_status = 'late_delivery' THEN 1 ELSE 0 END) as late
                
                FROM order_del
                GROUP BY delivered_date
                ) 

                SELECT
                    delivered_date,
                    early,
                    late,
                    (early + late) as all_delivery,
                    CASE 
                        WHEN (early + late) = 0 THEN 0
                        ELSE (early  * 1.0 / (early + late) ) * 100  -- Multiplicado por 1.0 para garantir divisão decimal
                    END AS early_del_perc,
                    CASE
                        WHEN (early + late) = 0 THEN 0
                        ELSE ( late * 1.0 / (early + late) ) * 100  -- Multiplicado por 1.0 para garantir divisão decimal
                    END AS late_del_perc

                FROM delivery_summary;
                """

#2.1:
early_late_del = pd.read_sql_query(early_late_del, engine)
early_late_del.drop_duplicates(inplace=True)
early_late_del.dropna(inplace=True)

print(early_late_del)


#%%

#2.2: a qtd de orders entregues antes e depois do prazo durante todo o periodo analisado :

del_count_orders_pie = """
                WITH order_del as (
                SELECT
                    order_id,
                    CASE 
                        WHEN order_estimated_delivery_date < order_delivered_customer_date
                            THEN 'late_del'
                        ELSE 'early_del'
                    END AS delivery_status
                FROM orders
                )

                SELECT
                    delivery_status,
                    count(*) as tot_orders
                FROM order_del
                GROUP BY delivery_status;
                """

del_count_orders_pie = pd.read_sql_query(del_count_orders_pie, engine)
del_count_orders_pie.drop_duplicates(inplace=True)
del_count_orders_pie.dropna(inplace=True)

print(del_count_orders_pie)

#%%

                                        # VISUALIZAÇAO GRAFICA DAS DUAS TABELAS ACIMA 

            #1: Formatando a area do grafico = 2 graficos e 1 texto :

                #1.1 = texto : (em cima no canto esquerdo) 
                        # com o numero total de order e o numero total de orders atrasadas 
            
                #1.2 = grafico de pizza : (em cima no canto direto)
                        # com a qtd de orders late e early destacando a qtd de orders atrasadas ;

                #1.3 = grafico de barras : (embaixo toda a extensao)
                        # na parte de baixo da area comparando entregas late x early por dia com as barras e
                            
                    #1.3.1: mostrando a qtd e as porcentagens no grafico de barras
                        
              
#1: Definindo a area de plotagem :
plt.figure(figsize=(24,18))
gs = gridspec.GridSpec(2, 2, width_ratios=[1,1], height_ratios=[0.5,1])


    #1.1: Fromatando a area do texto :
ax7 = plt.subplot(gs[0,0]) # 1a linha da 1a coluna da figura
ax7.axis('off') # remove as linhas de grade

        # Filtra o DataFrame para encontrar pedidos com status 'Late Delivery' e extrai o valor numerico da serie :
total_orders = del_count_orders_pie['tot_orders'].sum()
late_orders = (del_count_orders_pie.loc[del_count_orders_pie['delivery_status'] == 'late_del', 'tot_orders'] # Usa a máscara booleana(true ou False) para selecionar a coluna 'total_orders' apenas para as linhas onde o status é 'Late Delivery'. 
                .values[0]) # Extrai o valor numérico da Series resultante.
        # Insere o texto e centraliza
ax7.text(0.5, 0.5, f"Total de pedidos: {total_orders:,}\nTotal de pedidos atrasadas: {late_orders:,}", ha='center', va='center', fontsize=fonte_grande, wrap=True)


#del_count_orders_pie['delivery_status']
    #1.2: grafico de pizza : (em cima no canto direto)
ax8 = plt.subplot(gs[0,1]) # 1a linhda da 2a coluna da figura
labels = ["Atrasadas" if status == 'late_del' else "Antes do Prazo"
            for status in del_count_orders_pie['delivery_status'] ] # A list comprehension é projetada para ser declarativa. Ou seja, você descreve o que quer (a expressão) antes de especificar de onde vem os dados (o for).
sizes = del_count_orders_pie['tot_orders']
colors = ['#aadd77', '#ff6961'] # hex do verde pastel e vermelho pastel
explode = (0.1, 0)

ax8.pie(sizes, 
        labels=labels, 
        autopct='%1.1f%%', 
        startangle=140, 
        colors=colors, 
        explode=explode, 
        textprops={'fontsize': 14})

ax8.set_title("Comparação dos status das entregas: Antes x Depois do prazo ")
ax8.axis('equal')



    #1.3: Grafico de barras : (embaixo toda a extensao)
ax9 = plt.subplot(gs[1,:]) # 2a linha e todas as colunas da figura -- a linha 2 inteira
sns.barplot(x='delivered_date', y='all_delivery', data=early_late_del, color='#aadd77', label='Antes do Prazo', ax=ax9)
sns.barplot(x='delivered_date', y='late', data=early_late_del, color='#ff6961', label='Depois do prazo', ax=ax9)

        # Adicionando titulo e legendas no eixo x e y :
ax9.set_title("Entregas Totais vs. Entregas Atrasadas", fontsize=fonte_peq)
ax9.set_xlabel("Data de Entrega", fontsize=fonte_peq)
ax9.set_ylabel("Número de Entregas", fontsize=fonte_peq)
ax9.legend(fontsize='large')
    

    #1.3.1 : Adicionando o rotulo de % de pedidos entregues em cada barra :

            # DENTRO do prazo :
for bar, percent in zip( # combina cada barra (bar) com a respectiva porcentagem (percent)
                        ax9.patches[:len(early_late_del)] , # patches quer dizer cada retangulo do grafico // [:len(early_late_del)] garante que você pegue apenas as barras do primeiro conjunto de dados (entregas totais)
                        early_late_del['early_del_perc']):
        x_label_pos = bar.get_x() + bar.get_width() / 2 # Centraliza o rotulo no topo da barra
        y_label_pos = bar.get_height() + 50 # Retorna a altura da barra (valor no eixo Y) mais 50 pra cima
        ax9.text(
            x_label_pos, # posicao do eixo x
            y_label_pos, # posicao do eixo y
            f'{percent:.1f}%', # formata como a porcentagem vai aparecer no grafico
            ha='center', # centraliza horizontalmente o texto
            va='center', # centraliza verticalmente o texto
            fontsize=fonte_peq, # tamanho da fonte
            color='black') # cor do rotulo

            # FORA do prazo :
for bar, percent in zip(ax9.patches[len(early_late_del):], # [len(early_late_del):] = garante que você pegue apenas as barras do segundo conjunto de dados (entregas atrasadas)
                        early_late_del['late_del_perc']):
        x_label_pos = bar.get_x() + bar.get_width() / 2
        y_label_pos = bar.get_height() + 50
        ax9.text(
            x_label_pos,
            y_label_pos,
            f'{percent:.1f}%',
            ha='center',
            va='center',
            color='black')


plt.tight_layout()
plt.show()
