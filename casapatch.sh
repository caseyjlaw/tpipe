#! /bin/sh
 
casa=/home/casa/packages/RHEL5/release/casapy-41.0.24668-001-64b # customize this!
python=/users/claw/fasttransients/code # customize this!
 
cd $casa/lib64
 
# copy basic Python files
cp -a python2.6/casac.py python2.6/__casac__ $python
 
# copy dependent libraries, with moderate sophistication
for f in lib*.so* ; do
  if [ -h $f ] ; then
    cp -a $f $python/__casac__ # copy symlinks as links
  else
    case $f in
      *_debug.so) ;; # skip -- actually text files
      libgomp.*)
        # somehow patchelf fries this particular file
        cp -a $f $python/__casac__ ;;
      *)
        cp -a $f $python/__casac__
        patchelf --set-rpath '$ORIGIN' $python/__casac__/$f ;;
    esac
  fi
done
 
# patch rpaths of Python module binary files
cd $python/__casac__
for f in _*.so ; do
  patchelf --set-rpath '$ORIGIN' $f
done