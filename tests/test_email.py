from lh_tool.Email import CustomEmail


def test_email():
    print('Test lh_tool.Email')
    custom_email = CustomEmail(
        subject='Email: Information',
        content='This is a tests E-mail')
    custom_email.send()


if __name__ == '__main__':
    test_email()
