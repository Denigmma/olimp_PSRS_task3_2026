import os
import pandas as pd

from tel_cracker_opencl import crack_hashes

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
            for char in str(message).upper():
                if char in alphabet:
                    old_index = alphabet.find(char)
                    new_index = (old_index + bias) % len(alphabet)
                    decoded_message += alphabet[new_index]
                else:
                    decoded_message += char

            # эвристика для адресов
            if alphabet == alfavit_RU:
                if ('Д.' in decoded_message and 'КВ.' in decoded_message and
                    ('УЛ.' in decoded_message or 'ПЕР.' in decoded_message or 'ПР.' in decoded_message or
                     'МОСТ' in decoded_message or 'ПЛ.' in decoded_message)):
                    decrypted.append((message, decoded_message, t))
                    break

            # эвристика для e-mail
            if alphabet == alfavit_EU and key_words:
                if contains_key(decoded_message, key_words):
                    decrypted.append((message, decoded_message, t))
                    break

    return decrypted


def main():
    # email
    email_data = read_from_csv(input_file_email, 'email')
    decoded_emails = decryptor(email_data, alfavit_EU, possible_mail)

    # address
    address_data = read_from_csv(input_file_address, 'Адрес')
    decoded_addresses = decryptor(address_data, alfavit_RU)

    # tel (SHA1 -> phone) через OpenCL
    tel_hashes = read_from_csv(input_file_tel, 'Телефон')
    tel_hashes_norm = [str(h).strip().lower() for h in tel_hashes]

    tel_mapping = crack_hashes(
        tel_hashes_norm,
        phone_prefix_after_8="9",
        prefer_cpu=True
    )

    decoded_tels = []
    for h in tel_hashes_norm:
        decoded_tels.append((h, tel_mapping.get(h, ''), 'sha1_opencl(mask=89xxxxxxxxx; no_salt)'))

    # сборка финального датасета
    decoded_results = []
    n = max(len(decoded_emails), len(decoded_addresses), len(decoded_tels))
    for i in range(n):
        _, tel_decoded, tel_key = decoded_tels[i] if i < len(decoded_tels) else ('', '', '')
        _, email_decoded, email_key = decoded_emails[i] if i < len(decoded_emails) else ('', '', '')
        _, address_decoded, address_key = decoded_addresses[i] if i < len(decoded_addresses) else ('', '', '')
        decoded_results.append([tel_decoded, tel_key, email_decoded, email_key, address_decoded, address_key])

    write_to_csv(decoded_results)
    print(f"OK: saved to {output_file}")


if __name__ == "__main__":
    main()
