import os
import pandas as pd

alfavit_EU = 'ABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZ'
alfavit_RU = 'АБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯАБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ'


possible_mail = ['GMAIL.COM', 'YAHOO.COM', 'OUTLOOK.COM', 'HOTMAIL.COM', 'WISOZK.COM', 'GOODWIN.ORG',
                 'HAND.COM', 'GLOVER.COM', 'TORPHY.COM', 'FRIESEN.COM','JOHNSON.COM','FEENEY.COM',
                 'ABSHIRE.BIZ','ABSHIRE.BIZ','JOHNSON.COM','VOLKMAN.BIZ', 'MARQUARDT.BIZ', 'MOSCISKI.COM',
                 'JONES.ORG', 'LUBOWITZ.COM', 'SCHADEN.ORG', 'KUHIC.COM', 'SCHMITT.COM', 'WUNSCH.COM',
                 'EICHMANN.INFO', 'COLE.NET']

input_file_address = 'data/encoded_address_dataset.csv'
input_file_email = 'data/encoded_email_dataset.csv'
input_file_tel = 'data/encoded_tel_dataset.csv'
input_file_tel_result = 'data/result.txt'
output_file = 'decoded_dataset.csv'

def contains_key(string, keys):
    return any(key in string.upper() for key in keys)

def read_from_csv(input_file, column):
    df = pd.read_csv(input_file)
    return df[column].tolist()

def write_to_csv(decoded_data):
    file_exists = os.path.isfile(output_file)
    df = pd.DataFrame(decoded_data, columns=['tel', 'tel_key', 'email', 'email_key', 'address', 'address_key'])
    df.to_csv(output_file, mode='w', header=not file_exists, index=False)

def decryptor(messages, alphabet, key_words=None):
    decrypted = []
    for message in messages:
        for t in range(len(alphabet) // 2):
            bias = t
            decoded_message = ''
            for char in message.upper():
                if char in alphabet:
                    old_index = alphabet.find(char)
                    new_index = (old_index + bias) % len(alphabet)
                    decoded_message += alphabet[new_index]
                else:
                    decoded_message += char
            if alphabet == alfavit_EU:
                if contains_key(decoded_message, key_words):
                    decrypted.append((message, decoded_message, t))
                    break
            else:
                if ('Д.' in decoded_message and 'КВ.' in decoded_message and
                    ('УЛ.' in decoded_message or 'ПЕР.' in decoded_message or 'ПР.' in decoded_message or 'МОСТ' in decoded_message or 'ПЛ.' in decoded_message)):
                    decrypted.append((message, decoded_message, t))
                    break
    return decrypted

def read_tel_mapping(result_file):
    mapping = {}
    if os.path.isfile(result_file):
        with open(result_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if not line or ':' not in line:
                    continue
                h, tel = line.split(':', 1)
                mapping[h.strip().lower()] = tel.strip()
    return mapping

def main():
    email_data = read_from_csv(input_file_email, 'email')
    decoded_emails = decryptor(email_data, alfavit_EU, possible_mail)

    address_data = read_from_csv(input_file_address, 'Адрес')
    decoded_addresses = decryptor(address_data, alfavit_RU)

    tel_hashes = read_from_csv(input_file_tel, 'Телефон')
    tel_mapping = read_tel_mapping(input_file_tel_result)
    decoded_tels = []
    for h in tel_hashes:
        hh = str(h).strip().lower()
        decoded_tels.append((h, tel_mapping.get(hh, ''), 'sha1(mask=8xxxxxxxxxx; no_salt)'))

    decoded_results = []
    for i in range(max(len(decoded_emails), len(decoded_addresses), len(decoded_tels))):
        tel, tel_decoded, tel_key = decoded_tels[i] if i < len(decoded_tels) else ('', '', '')
        email, email_decoded, email_key = decoded_emails[i] if i < len(decoded_emails) else ('', '', '')
        address, address_decoded, address_key = decoded_addresses[i] if i < len(decoded_addresses) else ('', '', '')
        decoded_results.append([tel_decoded, tel_key, email_decoded, email_key, address_decoded, address_key])

    write_to_csv(decoded_results)

main()
