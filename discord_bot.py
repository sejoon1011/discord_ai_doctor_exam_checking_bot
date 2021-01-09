import asyncio
import discord
from classifier_model import main, classifier_model, preprocessing
import re
import numpy as np

client = discord.Client()

# 1-6에서 생성된 토큰을 이곳에 입력해주세요.
token = ""
model = None
# 봇이 구동되었을 때 동작되는 코드입니다.
@client.event
async def on_ready():
    global model
    model = main.main()
    game = discord.Game("공부")
    print("Logged in as ") #화면에 봇의 아이디, 닉네임이 출력됩니다.
    print(client.user.name)
    print(client.user.id)
    print("===========")
    # 디스코드에는 현재 본인이 어떤 게임을 플레이하는지 보여주는 기능이 있습니다.
    # 이 기능을 사용하여 봇의 상태를 간단하게 출력해줄 수 있습니다.
    await client.change_presence(status=discord.Status.online, activity=game)

# 봇이 새로운 메시지를 수신했을때 동작되는 코드입니다.
@client.event
async def on_message(message):
    chanel = message.channel
    match('99 609')
    if message.author.bot: #만약 메시지를 보낸사람이 봇일 경우에는
        return None #동작하지 않고 무시합니다.

    id = message.author.id #id라는 변수에는 메시지를 보낸사람의 ID를 담습니다.
    channel = message.channel #channel이라는 변수에는 메시지를 받은 채널의 ID를 담습니다.
    message = message.content
    if match(message): #만약 해당 메시지가 '!커맨드' 로 시작하는 경우에는
        data = split(message)
        predict = classifier_model.predict(model, data)
        Embed = discord.Embed(title="Prediction", description=predict, color=0x00ff21)
        await channel.send(predict)
    else: #위의 if에 해당되지 않는 경우
        #메시지를 보낸사람을 호출하며 말한 메시지 내용을 그대로 출력해줍니다.
        await channel.send("<@"+str(id) +">님이 \""+message+"\"라고 말하였습니다.")

def split(message):
    data = message.split()
    data = [float(i) for i in data]
    data = np.array(data).reshape(1, -1)
    data = preprocessing.st_trans(data)
    return data

def match(content):
    re1 = '[0-1][0-0][0-0] [0-7][0-0][0-0]'
    re2 = '[0-9][0-9] [0-6][0-9][0-9]'
    if re.match(re1, content) is not None or re.match(re2, content) is not None:
        return True
    else:
        return False


client.run(token)