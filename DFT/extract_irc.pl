#!/usr/bin/env perl

use warnings;
use strict;

my $pt = -1;
my $path = -1;
my $energy = 0;
my $converged = 0;

my $solv = shift;

while (<>) {
    if ( $pt == -1 and $_ =~ m/E\(RwB97XD\) =\s+(\S+)/ ) {
        print "$solv,0,0,$1,0\n";
    }

    if ( $_ =~ m/Corrected End Point Energy = (\S+)/ ) {
        $energy = $1;
    }

    if ( $_ =~ m/Delta-X Convergence Met/) {
        $converged = 1;
    }

    if ( $_ =~ m/Point Number:\s+(\d+)\s+Path Number:\s+(\d+)/) {
        $pt = $1;
        $path = $2;
    }

    if ( $_ =~ m/NET REACTION COORDINATE UP TO THIS POINT =\s+(\S+)/) {
        print "$solv,$pt,$path,$energy,$1\n";
        $converged = 0;
    }
}

