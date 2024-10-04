import random
import numpy as np
from faker import Faker
from random_username.generate import generate_username
import uuid
fake = Faker()

def get_name_obfuscation(name):
    name = name.lower()
    if np.random.uniform(0, 1) < 0.1:
        name = ' '.join([i.capitalize() for i in name.split()])

    if np.random.uniform(0, 1) < 0.1:
        name = name.split()[1] + ' ' + name.split()[0]

    if np.random.uniform(0, 1) < 0.1:
        name = name.split()[0][0] + ' ' + name.split()[1]
    
    if np.random.uniform(0, 1) < 0.5:
        name = name.replace(' ', '.')
    
    if np.random.uniform(0, 1) < 0.1:
        name += str(np.random.choice([i for i in range(100)] + [i for i in range(1950, 2020)]))

    if np.random.uniform(0, 1) < 0.1:
        name = generate_username()[0]
    return name

def get_student_name():
    name = fake.name()
    while len(name.split()) > 2:
        name = fake.name()
    return name

def get_student_email(name):
    name = get_name_obfuscation(name)
    email_start = ''.join(name.split())
    email_end = '@' + fake.ascii_free_email().split('@')[1]
    email = email_start + email_end
    return email

def get_student_phone():
    return fake.phone_number()

def get_student_address():
    address = fake.address()
    address = address.replace('\n', ', ')
    address = address.split(', ')
    address = ', '.join(address[:np.random.randint(1, len(address))])
    return address

def get_student_username(name):
    name = get_name_obfuscation(name)
    name = name.replace('.', '')
    name = name.replace(' ', '')
    name = name.lower()

    if np.random.uniform(0, 1) < 0.1:
        name += str(np.random.randint(0, 10000))

    # if np.random.uniform(0, 1) < 0.2:
    #     name = uuid.uuid4().hex
    return name

def get_student_id_num():
    id_num = random.choices([fake.ssn, fake.passport_number, fake.bban, fake.iban, fake.license_plate],
                             weights=[0.20, 0.20, 0.20, 0.20, 0.20], k=1)[0]()
    return id_num.replace('-', '')

def get_student_url(name):
    name = get_name_obfuscation(name)
    name = name.replace('.', '')
    name = name.lower()
    name = name.replace(' ', '')

    if np.random.uniform(0, 1) < 0.2:
        name = uuid.uuid4().hex
    
    social_media_platforms = {
        'LinkedIn': 'linkedin.com/in/',
        'YouTube': 'youtube.com/c/',
        'Instagram': 'instagram.com/',
        'GitHub': 'github.com/',
        'Facebook': 'facebook.com/',
        'Twitter': 'twitter.com/',
        'Coursera': 'www.coursera.org/user/',
    }
    platform, domain = random.choice(list(social_media_platforms.items()))
    fake_url = f'https://{domain}{name}'
    return fake_url

def get_student_data():
    name = get_student_name()
    email = get_student_email(name)
    username = get_student_username(name)
    id_num = get_student_id_num()
    phone = get_student_phone()
    url = get_student_url(name)
    address = get_student_address()

    id_num = id_num.replace(' ', '')
    phone = phone.replace('.', '-')
    
    # if np.random.uniform(0, 1) < 0.1:
    #     name = name.split()[0]
    if np.random.uniform(0, 1) < 0.5:
        email = None
    if np.random.uniform(0, 1) < 0.5:
        phone = None
    if np.random.uniform(0, 1) < 0.6:
        username = None
    if np.random.uniform(0, 1) < 0.6:
        id_num = None
    if np.random.uniform(0, 1) < 0.7:
        url = None
    if np.random.uniform(0, 1) < 0.7:
        address = None

    if bool(email) + bool(phone) + bool(url) + bool(address) >= 3:
        if np.random.uniform(0, 1) < 0.5:
            email = None
        if np.random.uniform(0, 1) < 0.5:
            phone = None
        if np.random.uniform(0, 1) < 0.5:
            username = None
        if np.random.uniform(0, 1) < 0.5:
            id_num = None
        if np.random.uniform(0, 1) < 0.5:
            url = None
        if np.random.uniform(0, 1) < 0.5:
            address = None
    
    data = {
        'NAME_STUDENT': name,
        'EMAIL': email,
        'USERNAME': username,
        'ID_NUM': None, # id_num
        'PHONE_NUM': None,
        'URL_PERSONAL': url,
        'STREET_ADDRESS': address,
    }

    
    return data

def get_prompt_heading(student_data):
    info = []
    names_for_prompt = {
        'NAME_STUDENT': 'Name',
        'EMAIL': 'email',
        'USERNAME': 'username',
        'ID_NUM': 'id_num',
        'PHONE_NUM': 'phone number',
        'URL_PERSONAL': 'url',
        'STREET_ADDRESS': 'address'
    }
    for key, value in student_data.items():
        if value:
            name = names_for_prompt[key]
            info.append(f'{name}: {value}')
    info = ', '.join(info)
    return info
    