import pandas as pd

urls = pd.read_csv('./data/fool_urls.csv', index_col = 0)

earnings_transcripts_links = []
for u in urls['urls']:
    if '/earnings/call-transcripts/' in u:
        earnings_transcripts_links.append(u)

print(earnings_transcripts_links)

links = pd.DataFrame(earnings_transcripts_links, columns=['earnings_links'])

links.to_csv('./data/earnings_links.csv')

# for u in urls['urls']:
#     if 'apple' in u:
#         print(u)