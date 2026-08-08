libpremia_path='../lib/libpremia.dll';
libpnl_path='../lib/libpnl.dll';
link(libpnl_path);
link(libpremia_path);
libpremiatb_path='../lib/libpremiatb.dll';
addinter(libpremiatb_path,'libpremiatb');
premia_init('../');
exec('./interface.sci');
printf("You can run ""premia()"" to launch the Premia interface\n");
