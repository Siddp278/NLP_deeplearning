# Install googletrans alpha version, it works properly.
import googletrans
from googletrans import Translator

print(googletrans.LANGUAGES) # gives the dictionary of the language it supports and its codes.
tra = Translator(service_urls=['translate.googleapis.com'])
result = tra.translate('アイヴァンモリスの翻訳のこの抜粋で、セイは宮廷生活のドラマを垣間見ることができます。彼女は、猫の命婦夫人に対する悲劇的な誤解が、' 
        '法廷でお気に入りの犬であるオキナマロをどのように支持から失ったかについて語っています。花輪と花で覆われた後、彼は殴打されて追放さ'
        'れ、この運命のねじれを元に戻して好意を取り戻すのに苦労しました。')

print(result.src)
print(result.dest)
# print(result.origin)
print(type(result.text))
print(result.pronunciation)
