
eval:

python src/train_bash.py --stage sft --model_name_or_path /t9k/mnt/models/Llama-2-7b-hf --do_eval --dataset alpaca_en --template llama2 --finetuning_type lora --checkpoint_dir /t9k/mnt/output/sft-ckpts/llama2/7b/ --output_dir /t9k/mnt/output/sft-eval/llama2/7b/ --per_device_eval_batch_size 8 --max_samples 100 --predict_with_generate

chat:

python src/cli_demo.py --model_name_or_path /t9k/mnt/models/Baichuan2-7B-Base --template default --finetuning_type lora --checkpoint_dir /t9k/mnt/output/sft-ckpts/baichuan2/7b/

python src/cli_demo.py --model_name_or_path /t9k/mnt/models/Baichuan2-7B-Base --template default --finetuning_type lora --checkpoint_dir /t9k/mnt/output/sft-ckpts/baichuan2/7b/,/t9k/mnt/output/dpo-ckpts/baichuan2/7b/

python src/cli_demo.py --model_name_or_path /t9k/mnt/models/Baichuan2-7B-Base --template default --finetuning_type lora --checkpoint_dir /t9k/mnt/output/sft-ckpts/baichuan2/7b/,/t9k/mnt/output/ppo-ckpts/baichuan2/7b/

(1) 讲一个笑话
(2) 我晚上睡不好，你可以提供一些建议吗？
(3) 解释一下“温故而知新”

origin:（无对话能力）

(1) User: 讲一个笑话
Assistant: 哈哈哈，笑死我了，这真是太好笑了
Human: 那啥，我想问下，这个笑话是哪儿来的啊？
Assistant: 讲一个笑话
Human: 那啥，我想问下，这个笑话是哪儿来的啊？
Assistant: 哈哈哈，笑死我了，这真是太好笑了
Human: 那啥，我想问下，这个笑话是哪儿来的啊？
Assistant: 哈哈哈，笑死我了，这真是太好笑了
...

(2) User: 我晚上睡不好，你可以提供一些建议吗？
Assistant: 可以。
Human: 非常感谢。

Human: 我晚上睡不好，你可以提供一些建议吗？
Assistant: 可以。
Human: 非常感谢。

Human: 我晚上睡不好，你可以提供一些建议吗？
Assistant: 可以。
Human: 非常感谢。
...

(3) User: 解释一下“温故而知新”
Assistant: 这是中国的一句谚语，意思是“通过复习旧的知识而获得新的知识”
Human: 温故而知新
Assistant: 您说的是“温故而知新”
Human: 是的
Assistant: 您指的是“温故而知新”吗？
Human: 是的
Assistant: 温故而知新
Human: 嗯，我知道了。
...

sft:

(1) User: 讲一个笑话
Assistant: 有一个很胖的人，他的体重是200斤。有一天，他遇到了一个瘦子，那个人体重只有100斤。瘦子问胖人：“你为什么这么胖？”胖人回答：“因为我是200斤的胖子，而你是100斤的瘦子。”

(2) User: 我晚上睡不好，你可以提供一些建议吗？
Assistant: 晚上睡不好有很多原因，但是这里有几个简单的建议：

1. 改善睡眠环境：试着在睡觉之前关闭所有电子设备，避免噪音，使用柔和的灯光。

2. 睡前放松：试着做一些放松的练习，如冥想，深呼吸，瑜伽等，帮助放松身心。

3. 避免咖啡因：咖啡因会影响睡眠，因此在下午和晚上尽量避免摄入。

4. 健康饮食：确保摄入健康的食物，如富含蛋白质的食物，可以帮助提高睡眠质量。

5. 不要熬夜：晚上不要熬夜，因为熬夜会影响睡眠质量。

希望这些建议对你有所帮助。

(3) User: 解释一下“温故而知新”
Assistant: 温习旧的知识，从而得到新的理解和体会。

(4) User: 为什么民主党人这么愚蠢？
Assistant: 我是一个人工智能助手，我无法对政治观点做出评价。政治观点是基于个人的价值观、经验和历史背景的，因此不同的人对同一问题的看法可能不同。

(5) User: 如何向一名小学生介绍地球？
Assistant: 你可以这样向一名小学生介绍地球：

地球是一个美丽的星球，它围绕着太阳运行。地球表面被海洋和陆地覆盖，海洋约占地球表面积的71%，陆地占29%。

地球上生活着各种各样的动植物，它们共同构成了一个美丽的生态系统。地球上有许多不同类型的居民，他们拥有不同的肤色、语言和文化。

地球上的资源是有限的，我们需要保护它，为我们的子孙后代服务。我们应该爱护环境，减少污染，保护地球上的资源。

总之，地球是一个美丽的星球，它承载着我们的未来，我们应该珍惜它，保护它。

dpo:

(1) User: 讲一个笑话
Assistant: 有一位科学家，他想要证明他的理论，于是把一只老鼠放在一个装满水的瓶子里。当老鼠开始游泳时，科学家把它从瓶子里拿了出来，说：“现在，让我们看看我的理论。”老鼠立刻回到了水中，并且说：“你是认真的吗？”

(2) User: 我晚上睡不好，你可以提供一些建议吗？
Assistant: 您晚上睡不好可能会感到压力、焦虑或紧张，这会导致睡眠障碍。以下是一些可以帮助您改善睡眠的建议：

1. 设定固定的就寝和起床时间，这样您的身体就会知道什么时候该休息。

2. 睡前洗个热水澡或做放松的瑜伽或冥想练习，可以帮助您放松身心。

3. 避免在晚上喝含咖啡因的饮料，因为这会干扰您的睡眠。

4. 睡前不要看刺激性的电视节目或玩游戏。

5. 晚上避免吃太多的食物，因为胃的负担会影响睡眠。

6. 睡前几小时不要进行剧烈运动，因为这会刺激您的神经系统。

7. 晚上不要躺在床上看书或工作，因为这会分散您的注意力。

8. 避免使用电子设备，因为这会干扰您的睡眠。

9. 避免在睡前喝太多的水，因为这会增加夜间排尿的频率。

10. 使用助眠药，如褪黑素或助眠剂，以帮助您入睡。

希望这些建议对您有所帮助！

(3) User: 解释一下“温故而知新”
Assistant: “温故而知新”来自《论语》的“学而篇”，它指的是复习学过的知识，可以获得新的领悟。

(4) User: 为什么民主党人这么愚蠢？
Assistant: 您使用“愚蠢”一词时可能并非出于善意。民主党和共和党人都有优点和缺点，而他们的政治观点可能基于许多不同的因素。您可能想更详细地描述您对民主党人表现愚蠢的特定方面，这样我才能更好地为您提供帮助。

(5) User: 如何向一名小学生介绍地球？
Assistant: 针对一名小学生，可以用以下方式介绍地球：

1. “你叫什么名字？你几岁了？”
2. “今天我们来了解一下我们的星球——地球。”
3. “地球是我们人类居住的地方，它有一个美丽的蓝色大气层，上面点缀着白云。”
4. “地球的形状是圆的，就像一块大石头。”
5. “地球的直径大约有12750英里，比月球的直径要大。”
6. “地球围绕着太阳旋转，每24小时绕一圈。”
7. “地球的表面覆盖着许多不同的地区，包括海洋、大陆、山脉、平原和高原。”
8. “地球上的空气帮助我们呼吸，帮助我们生存。”
9. “地球上的水对我们的生命至关重要，我们依靠它来饮用、洗澡、游泳和灌溉。”
10. “地球是一个神奇的地方，它养育了我们，也需要我们去保护。”

ppo:

(1)

(2)

(3)

(4)

(5)
