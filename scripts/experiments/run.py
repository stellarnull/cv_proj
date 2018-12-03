#! /usr/bin/python

import sys, os, argparse;

class Options_Run():
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="parser for running script")
        subparsers = self.parser.add_subparsers(title="subcommands", dest="subcommand")

        # ns args
        ns_arg = subparsers.add_parser("ns",
                                    help="parser for ns arguments")
        ns_arg.add_argument("--style", type=int, default=9,
                                help="9 or 21 styles")

        # msg args
        msg_arg = subparsers.add_parser("msg",
                                    help="parser for msg arguments")
        msg_arg.add_argument("--style", type=int, default=9,
                                help="9 or 21 styles")


    def parse(self):
        return self.parser.parse_args()


if __name__ == "__main__":
    args = Options_Run().parse()
    if args.subcommand is None:
        raise ValueError("ERROR: specify the algorithm to be used")
    if args.style != 9 and args.style != 21:
        raise ValueError("ERROR: wrong style catagory")
    style_dir_name = "images/" + str(args.style) + "styles/"
    content_dir_name = "images/content/"
    style_dir_list =  os.listdir(style_dir_name)
    content_dir_list =  os.listdir(content_dir_name)
    for style_file in style_dir_list:
        style_path = os.path.join('%s%s' % (style_dir_name, style_file))
        for content_file in content_dir_list:
            content_path = os.path.join('%s%s' % (content_dir_name, content_file))
            if args.subcommand == "ns":
                cmd = "python main.py optim --content-image " + content_path + " --style-image " + style_path + " --output-image images/outputs/" + content_file[:-4] + "_" + style_file[:-4] + "_ns.jpg"
            elif args.subcommand == "msg":
                cmd = "python main.py eval --model models/21styles.model --content-image " + content_path + " --style-image " + style_path + " --output-image images/outputs/" + content_file[:-4] + "_" + style_file[:-4] + "_msg.jpg"
            print(cmd)
            os.system(cmd)




