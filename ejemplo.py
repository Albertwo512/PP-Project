from ftplib import FTP

host = 'ftp.b2studios.com.mx'
user = 'alberto@b2studios.com.mx'
password = 'AlbertoA@512'

try:
    ftp = FTP(host,user,password)
    print('exito al 100')
    print('\n Nos encontramos en la carpeta ' +ftp.pdw()+'\n')

    ftp.dir()

    ftp.cwd('public_html')

except Exception as e:
    print('conexion errada: '+str(e))