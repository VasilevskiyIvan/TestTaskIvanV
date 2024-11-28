import asyncio
import logging
import os
from dotenv import load_dotenv
from aiogram import Bot, Dispatcher, types
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.filters import Command
from aiogram import Router
from PIL import Image
import torch
from torchvision import models, transforms
from models import MyCNN, MyResNet, MyResNet2

logging.basicConfig(level=logging.INFO)

load_dotenv()
bot = Bot(os.getenv('TOKEN'))

storage = MemoryStorage()
dp = Dispatcher(storage=storage)
router = Router()

# Пути к весам моделей
resnet_model_path = "resnet34_w.pth"
mycnn_model_path = "MyCNN_w2.pth"
myresnet_model_path = "MyResNet_w1.pth"
myresnet2_model_path = "MyResNet_w2.pth"

# Классы
class_names = ['гусь', 'индюк', 'курица', 'петух', 'страус', 'утка', 'цыпленок']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Инициализация трансформации для изображений
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@router.message(Command("start"))
async def start_handler(message: types.Message):
    markup = types.ReplyKeyboardMarkup(
        keyboard=[
            [types.KeyboardButton(text="Собственная CNN")],
            [types.KeyboardButton(text="Дообученный ResNet")],
            [types.KeyboardButton(text="Собственный аналог ResNet (вариант 1)")],
            [types.KeyboardButton(text="Собственный аналог ResNet (вариант 2)")]
        ],
        resize_keyboard=True
    )
    await message.reply("Привет! Выберите модель для классификации:", reply_markup=markup)

@router.message(lambda message: message.text in ["Собственная CNN", "Дообученный ResNet", "Собственный аналог ResNet (вариант 1)", "Собственный аналог ResNet (вариант 2)"])
async def model_selection_handler(message: types.Message):
    global model, transform
    selected_model = message.text

    if selected_model == "Собственная CNN":
        model = MyCNN(len(class_names))
        model.load_state_dict(torch.load(mycnn_model_path, map_location=device))
        model = model.to(device)
        model.eval()
        await message.reply("Вы выбрали модель: Собственная CNN."
                            "Точность около 95%, но модель чувствительна к фону. Точность на оригинальном тесте около 0.95, точность на новых данных около 0.875")

    elif selected_model == "Дообученный ResNet":
        model = models.resnet34(pretrained=False)
        num_features = model.fc.in_features
        model.fc = torch.nn.Linear(num_features, len(class_names))
        model.load_state_dict(torch.load(resnet_model_path, map_location=device))
        model = model.to(device)
        model.eval()
        await message.reply("Вы выбрали модель: Дообученный ResNet."
                            "Точность около 99% с минимальными ошибками классификации. Точность на оригинальном тесте 1.00, точность на новых данных >0.97")

    elif selected_model == "Собственный аналог ResNet (вариант 1)":
        model = MyResNet(len(class_names))
        model.load_state_dict(torch.load(myresnet_model_path, map_location=device))
        model = model.to(device)
        model.eval()
        await message.reply("Вы выбрали модель: Собственный аналог ResNet (вариант 1)"
                            "Точность около 95%, но возможны ошибки на схожих видах птиц. Точность на оригинальном тесте >0.95, точность на новых данных >0.75 (в среднем около 0.85)")

    elif selected_model == "Собственный аналог ResNet (вариант 2)":
        model = MyResNet2(len(class_names))  # Используем новый класс
        model.load_state_dict(torch.load(myresnet2_model_path, map_location=device))
        model = model.to(device)
        model.eval()
        await message.reply("Вы выбрали модель: Собственный аналог ResNet (вариант 2)"
                            "Эта модель отличается улучшенной обработкой мелких деталей. Точность на оригинальном тесте >0.95, точность на новых данных >0.8")

    markup = types.ReplyKeyboardRemove()
    await message.reply("Теперь отправьте мне фото для классификации.", reply_markup=markup)


# Обработчик фотографий
@router.message(lambda message: message.photo)
async def photo_handler(message: types.Message):
    photo = message.photo[-1]
    file = await bot.download(photo.file_id)

    image = Image.open(file).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        output = model(input_tensor)
        _, predicted_idx = torch.max(output, 1)
        predicted_class = class_names[predicted_idx.item()]

    await message.reply(f"Птица на данной фотографии относится к классу: *{predicted_class}*", parse_mode="Markdown")

dp.include_router(router)


async def main():
    try:
        await dp.start_polling(bot)
    finally:
        await bot.session.close()

try:
    asyncio.run(main())
except RuntimeError:
    import nest_asyncio

    nest_asyncio.apply()
    asyncio.get_event_loop().run_until_complete(main())
